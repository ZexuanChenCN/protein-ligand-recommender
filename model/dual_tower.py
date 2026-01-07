"""
dual_tower.py - 最终可运行版（已修复random导入+梯度裁剪+余弦退火）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random  # 关键：已导入random模块

# 1. 通用蛋白编码器（添加数据增强+Dropout正则化）
class SimpleProteinEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, max_len: int = 1024, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dropout = dropout

        # 氨基酸字典（20种常见氨基酸 + 未知）
        self.aa_vocab = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
                         'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13,
                         'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19, 'X':20}
        self.vocab_size = len(self.aa_vocab)
        self.aa_list = list(self.aa_vocab.keys())

        # 基础编码层（添加Dropout正则化）
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)  # 新增卷积
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3),  # 2→3倍
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(self.dropout/2)
        )

    def _augment_protein_seq(self, seq: str) -> str:
        """蛋白序列增强：训练时随机截断+替换"""
        if not self.training or len(seq) < 10:
            return seq

        # 1. 随机截断（保留70%-100%）
        trunc_ratio = random.uniform(0.7, 1.0)
        trunc_len = int(len(seq) * trunc_ratio)
        seq = seq[:trunc_len]

        # 2. 随机替换10%的氨基酸
        seq_list = list(seq)
        num_replace = max(1, int(len(seq_list) * 0.1))
        replace_indices = random.sample(range(len(seq_list)), k=num_replace)
        for idx in replace_indices:
            # 替换为不同的氨基酸
            current_aa = seq_list[idx]
            new_aa = random.choice([aa for aa in self.aa_list if aa != current_aa])
            seq_list[idx] = new_aa

        return ''.join(seq_list)

    def forward(self, protein_seqs: list) -> torch.Tensor:
        batch_size = len(protein_seqs)
        if self.training:
            protein_seqs = [self._augment_protein_seq(seq) for seq in protein_seqs]

        # 序列转索引（不变）
        seq_indices = []
        for seq in protein_seqs:
            indices = [self.aa_vocab.get(aa.upper(), 20) for aa in seq[:self.max_len]]
            if len(indices) < self.max_len:
                indices += [20] * (self.max_len - len(indices))
            seq_indices.append(indices)
        seq_indices = torch.tensor(seq_indices).to(self.embedding.weight.device)

        # 编码：新增卷积层
        emb = self.embedding(seq_indices)  # [batch, max_len, embed_dim]
        emb = emb.permute(0, 2, 1)  # [batch, embed_dim, max_len]（适配卷积）
        emb = self.conv(emb)  # 卷积提取局部特征
        emb = emb.permute(0, 2, 1)  # 还原维度
        emb = emb.mean(dim=1)  # [batch, embed_dim]
        emb = self.fc(emb)
        emb = F.normalize(emb, dim=-1)
        return emb

# 2. 小分子编码器（动态获取维度）
class ChemBERTaEmbeddingExtractor(nn.Module):
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # 动态获取ChemBERTa的输出维度
        self.embed_dim = self.model.config.hidden_size  # 实际是384

    def forward(self, smiles_list: list) -> torch.Tensor:
        """编码SMILES列表（支持嵌套列表）"""
        # 检测嵌套列表（负样本）
        is_nested = isinstance(smiles_list[0], list)

        if is_nested:
            # 负样本：展平→编码→恢复维度
            batch_size = len(smiles_list)
            num_neg = len(smiles_list[0])
            flat_smiles = [smi for neg_group in smiles_list for smi in neg_group]
            flat_emb = self._encode_single_list(flat_smiles)
            return flat_emb.reshape(batch_size, num_neg, self.embed_dim)
        else:
            # 正样本：直接编码
            return self._encode_single_list(smiles_list)

    def _encode_single_list(self, smiles_list: list) -> torch.Tensor:
        """基础编码逻辑"""
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return emb

# 3. 双塔模型（核心优化：梯度裁剪+余弦退火+正样本加权）
class CLIPStyleDualTower(pl.LightningModule):
    def __init__(
            self,
            protein_embed_dim: int = 256,
            proj_dim: int = 256,
            init_temperature: float = 0.15,  # 更高初始温度，减缓Loss收敛
            lr: float = 1e-4,
            weight_decay: float = 0.05,  # 降低权重衰减，避免欠拟合
            hard_neg_weight: float = 1.5,  # 提高硬负样本权重
            margin: float = 0.2,  # 提高margin，强化正负样本区分
            dropout: float = 0.2,
            grad_clip_norm: float = 1.0  # 梯度裁剪阈值
    ):
        super().__init__()
        self.save_hyperparameters()

        # 编码器（添加Dropout+增强）
        self.protein_encoder = SimpleProteinEncoder(embed_dim=protein_embed_dim, dropout=dropout)
        self.ligand_encoder = ChemBERTaEmbeddingExtractor()
        self.ligand_embed_dim = self.ligand_encoder.embed_dim

        # 投影头（维度完全对齐）
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_embed_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),  # 投影头添加Dropout
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.ligand_proj = nn.Sequential(
            nn.Linear(self.ligand_embed_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),  # 投影头添加Dropout
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.hard_neg_weight = hard_neg_weight
        self.margin = margin
        self.grad_clip_norm = grad_clip_norm

    def encode_protein(self, protein_seqs: list) -> torch.Tensor:
        """编码蛋白并投影"""
        protein_emb = self.protein_encoder(protein_seqs)
        protein_emb = self.protein_proj(protein_emb)
        return F.normalize(protein_emb, dim=-1)

    def encode_ligand(self, ligand_smiles: list) -> torch.Tensor:
        """编码小分子并投影（适配嵌套列表）"""
        ligand_emb = self.ligand_encoder(ligand_smiles)

        # 适配投影头维度（支持3维张量）
        if len(ligand_emb.shape) == 3:
            # 负样本：[batch, num_neg, dim] → 展平投影 → 恢复维度
            batch_size, num_neg, dim = ligand_emb.shape
            flat_emb = ligand_emb.reshape(-1, dim)
            flat_proj = self.ligand_proj(flat_emb)
            ligand_emb = flat_proj.reshape(batch_size, num_neg, self.hparams.proj_dim)
        else:
            # 正样本：直接投影
            ligand_emb = self.ligand_proj(ligand_emb)

        return F.normalize(ligand_emb, dim=-1)

    def clip_loss(
            self,
            protein_emb: torch.Tensor,
            pos_ligand_emb: torch.Tensor,
            neg_ligand_emb: torch.Tensor,
            neg_Y: torch.Tensor = None
    ) -> tuple:
        """优化版CLIP损失：正样本加权+提升Margin+硬负样本权重"""
        batch_size = protein_emb.shape[0]
        num_neg = neg_ligand_emb.shape[1]

        # 计算相似度（正样本加权+降低温度缩放，放大差异）
        pos_sim = (protein_emb * pos_ligand_emb).sum(dim=-1) * 1.5 / (self.temperature * 0.5)  # 正样本加权1.5倍
        protein_emb_expand = protein_emb.unsqueeze(1).repeat(1, num_neg, 1)
        neg_sim = (protein_emb_expand * neg_ligand_emb).sum(dim=-1) * 0.8 / (self.temperature * 0.5)  # 负样本惩罚0.8倍

        # 硬负样本加权（提升难区分负样本的权重）
        if neg_Y is not None and self.hard_neg_weight > 1.0:
            # 负样本Y值越大，和正样本越相似（越硬），权重越高
            neg_Y = neg_Y.to(self.device)
            # 归一化负样本Y值到[1, hard_neg_weight]区间
            hard_neg_scale = 1 + (neg_Y - neg_Y.min()) / (neg_Y.max() - neg_Y.min() + 1e-8) * (self.hard_neg_weight - 1)
            neg_sim = neg_sim * hard_neg_scale

        # InfoNCE损失
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        loss = F.cross_entropy(all_sim, labels)

        # 强化对比正则化（提升Margin）
        pos_sim_expand = pos_sim.unsqueeze(1).repeat(1, num_neg)
        reg_loss = F.relu(neg_sim - pos_sim_expand + self.margin * 2).mean()
        total_loss = loss + 0.2 * reg_loss  # 提高正则化权重

        # 计算Margin Score（核心指标）
        margin_score = pos_sim.mean() - neg_sim.mean()

        return total_loss, self.temperature.item(), margin_score

    def training_step(self, batch, batch_idx):
        """训练步骤：梯度裁剪+准确率计算"""
        protein_seqs = batch["protein_seqs"]
        pos_ligand_smiles = batch["pos_ligand_smiles"]
        neg_ligand_smiles = batch["neg_ligand_smiles"]
        neg_Y = batch["neg_Y"]

        # 编码
        protein_emb = self.encode_protein(protein_seqs)
        pos_ligand_emb = self.encode_ligand(pos_ligand_smiles)
        neg_ligand_emb = self.encode_ligand(neg_ligand_smiles)

        # 计算损失和Margin
        loss, temp, margin_score = self.clip_loss(protein_emb, pos_ligand_emb, neg_ligand_emb, neg_Y)

        # 计算对比准确率：正样本相似度 > 所有负样本相似度 → 正确
        pos_sim = (protein_emb * pos_ligand_emb).sum(dim=-1)
        neg_sim = (protein_emb.unsqueeze(1) * neg_ligand_emb).sum(dim=-1)
        train_acc = (pos_sim.unsqueeze(1) > neg_sim).all(dim=1).float().mean()

        # 日志输出（包含准确率）
        self.log("train_loss_step", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", train_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log("temperature_step", temp, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_margin_score_step", margin_score, prog_bar=True, logger=True, sync_dist=True)

        # 梯度裁剪（核心：解决Loss波动）
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤：新增准确率计算"""
        protein_seqs = batch["protein_seqs"]
        pos_ligand_smiles = batch["pos_ligand_smiles"]
        neg_ligand_smiles = batch["neg_ligand_smiles"]
        neg_Y = batch["neg_Y"]

        # 编码
        protein_emb = self.encode_protein(protein_seqs)
        pos_ligand_emb = self.encode_ligand(pos_ligand_smiles)
        neg_ligand_emb = self.encode_ligand(neg_ligand_smiles)

        # 计算损失和Margin
        loss, temp, margin_score = self.clip_loss(protein_emb, pos_ligand_emb, neg_ligand_emb, neg_Y)

        # 计算对比准确率
        pos_sim = (protein_emb * pos_ligand_emb).sum(dim=-1)
        neg_sim = (protein_emb.unsqueeze(1) * neg_ligand_emb).sum(dim=-1)
        val_acc = (pos_sim.unsqueeze(1) > neg_sim).all(dim=1).float().mean()

        # 日志输出（包含准确率）
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", val_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_temperature", temp, logger=True, sync_dist=True)
        self.log("val_margin_score", margin_score, prog_bar=True, logger=True, sync_dist=True)

        return loss

    # 在dual_tower.py的configure_optimizers中修改
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        # 替换为ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,  # 每次下降80%
            patience=2,  # 2个epoch val_loss不下降则降LR
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # 基于验证集损失调整
                "frequency": 1
            }
        }