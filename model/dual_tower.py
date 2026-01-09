import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
import random
from saprot.base import SaprotBaseModel

class SaProtProteinEncoder(nn.Module):
    def __init__(self,
                 saprot_config_path: str = "./saprot_base",
                 embed_dim: int = 256,
                 freeze_backbone: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.saprot = SaprotBaseModel(
            task="base",
            config_path=saprot_config_path,
            load_pretrained=True,
            freeze_backbone=freeze_backbone,
            use_lora=False,
            lr_scheduler_kwargs={"init_lr": 1e-5},
            optimizer_kwargs={"weight_decay": 0.01},
            save_path=None,
            from_checkpoint=None,
            load_prev_scheduler=False,
            save_weights_only=True
        )

        self.saprot.initialize_model()
        self.saprot_tokenizer = self.saprot.tokenizer
        self.saprot_model = self.saprot.model


        self.saprot_hidden_dim = self.saprot_model.config.hidden_size

        self.proj_layer = nn.Sequential(
            nn.Linear(self.saprot_hidden_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout/2)
        )

    def _augment_protein_seq(self, seq: str) -> str:
        if not self.training or len(seq) < 10:
            return seq

        trunc_ratio = random.uniform(0.7, 1.0)
        trunc_len = int(len(seq) * trunc_ratio)
        seq = seq[:trunc_len]

        seq_list = list(seq)
        num_replace = max(1, int(len(seq_list) * 0.1))
        replace_indices = random.sample(range(len(seq_list)), k=num_replace)
        aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        for idx in replace_indices:
            current_aa = seq_list[idx]
            new_aa = random.choice([aa for aa in aa_list if aa != current_aa])
            seq_list[idx] = new_aa

        return ''.join(seq_list)

    def forward(self, protein_seqs: list) -> torch.Tensor:
        """
        用SaProt编码蛋白序列，输出256维embedding
        Args:
            protein_seqs: 蛋白序列列表（如["MAKGEIKAAL", "VHLTPEEKSAV"]）
        Returns:
            [batch, 256] 归一化后的embedding
        """
        batch_size = len(protein_seqs)

        if self.training:
            protein_seqs = [self._augment_protein_seq(seq) for seq in protein_seqs]

        inputs = self.saprot_tokenizer(
            protein_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.saprot_model.device)

        with torch.no_grad() if self.saprot.freeze_backbone else torch.enable_grad():
            hidden_repr = self.saprot.get_hidden_states(inputs, reduction="mean")  # [batch, 1280]

        protein_emb = torch.stack(hidden_repr)  # [batch, 1280]
        protein_emb = self.proj_layer(protein_emb)  # [batch, 256]
        protein_emb = F.normalize(protein_emb, dim=-1)

        return protein_emb

class ChemBERTaEmbeddingExtractor(nn.Module):
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size  #

    def forward(self, smiles_list: list) -> torch.Tensor:

        is_nested = isinstance(smiles_list[0], list)

        if is_nested:
            batch_size = len(smiles_list)
            num_neg = len(smiles_list[0])
            flat_smiles = [smi for neg_group in smiles_list for smi in neg_group]
            flat_emb = self._encode_single_list(flat_smiles)
            return flat_emb.reshape(batch_size, num_neg, self.embed_dim)
        else:

            return self._encode_single_list(smiles_list)

    def _encode_single_list(self, smiles_list: list) -> torch.Tensor:
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


class CLIPStyleDualTower(pl.LightningModule):
    def __init__(
            self,
            saprot_config_path: str = "./saprot_base",
            protein_embed_dim: int = 256,
            proj_dim: int = 256,
            init_temperature: float = 0.15,
            lr: float = 1e-4,
            weight_decay: float = 0.05,
            hard_neg_weight: float = 1.5,
            margin: float = 0.2,
            dropout: float = 0.2,
            grad_clip_norm: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.protein_encoder = SaProtProteinEncoder(
            saprot_config_path=saprot_config_path,
            embed_dim=protein_embed_dim,
            freeze_backbone=True,
            dropout=dropout
        )
        self.ligand_encoder = ChemBERTaEmbeddingExtractor()
        self.ligand_embed_dim = self.ligand_encoder.embed_dim


        self.protein_proj = nn.Sequential(
            nn.Linear(protein_embed_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.ligand_proj = nn.Sequential(
            nn.Linear(self.ligand_embed_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.hard_neg_weight = hard_neg_weight
        self.margin = margin
        self.grad_clip_norm = grad_clip_norm

    def encode_protein(self, protein_seqs: list) -> torch.Tensor:
        protein_emb = self.protein_encoder(protein_seqs)
        protein_emb = self.protein_proj(protein_emb)
        return F.normalize(protein_emb, dim=-1)

    def encode_ligand(self, ligand_smiles: list) -> torch.Tensor:
        ligand_emb = self.ligand_encoder(ligand_smiles)
        if len(ligand_emb.shape) == 3:
            batch_size, num_neg, dim = ligand_emb.shape
            flat_emb = ligand_emb.reshape(-1, dim)
            flat_proj = self.ligand_proj(flat_emb)
            ligand_emb = flat_proj.reshape(batch_size, num_neg, self.hparams.proj_dim)
        else:
            ligand_emb = self.ligand_proj(ligand_emb)

        return F.normalize(ligand_emb, dim=-1)

    def clip_loss(
            self,
            protein_emb: torch.Tensor,
            pos_ligand_emb: torch.Tensor,
            neg_ligand_emb: torch.Tensor,
            neg_Y: torch.Tensor = None
    ) -> tuple:
        batch_size = protein_emb.shape[0]
        num_neg = neg_ligand_emb.shape[1]

        pos_sim = (protein_emb * pos_ligand_emb).sum(dim=-1) * 1.5 / (self.temperature * 0.5)
        protein_emb_expand = protein_emb.unsqueeze(1).repeat(1, num_neg, 1)
        neg_sim = (protein_emb_expand * neg_ligand_emb).sum(dim=-1) * 0.8 / (self.temperature * 0.5)

        if neg_Y is not None and self.hard_neg_weight > 1.0:
            neg_Y = neg_Y.to(self.device)
            hard_neg_scale = 1 + (neg_Y - neg_Y.min()) / (neg_Y.max() - neg_Y.min() + 1e-8) * (self.hard_neg_weight - 1)
            neg_sim = neg_sim * hard_neg_scale

        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        loss = F.cross_entropy(all_sim, labels)
        pos_sim_expand = pos_sim.unsqueeze(1).repeat(1, num_neg)
        reg_loss = F.relu(neg_sim - pos_sim_expand + self.margin * 2).mean()
        total_loss = loss + 0.2 * reg_loss
        margin_score = pos_sim.mean() - neg_sim.mean()
        return total_loss, self.temperature.item(), margin_score

    def training_step(self, batch, batch_idx):
        protein_seqs = batch["protein_seqs"]
        pos_ligand_smiles = batch["pos_ligand_smiles"]
        neg_ligand_smiles = batch["neg_ligand_smiles"]
        neg_Y = batch["neg_Y"]
        protein_emb = self.encode_protein(protein_seqs)
        pos_ligand_emb = self.encode_ligand(pos_ligand_smiles)
        neg_ligand_emb = self.encode_ligand(neg_ligand_smiles)
        loss, temp, margin_score = self.clip_loss(protein_emb, pos_ligand_emb, neg_ligand_emb, neg_Y)
        pos_sim = (protein_emb * pos_ligand_emb).sum(dim=-1)
        neg_sim = (protein_emb.unsqueeze(1) * neg_ligand_emb).sum(dim=-1)
        train_acc = (pos_sim.unsqueeze(1) > neg_sim).all(dim=1).float().mean()

        self.log("train_loss_step", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", train_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log("temperature_step", temp, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_margin_score_step", margin_score, prog_bar=True, logger=True, sync_dist=True)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)

        return loss

    def validation_step(self, batch, batch_idx):
        protein_seqs = batch["protein_seqs"]
        pos_ligand_smiles = batch["pos_ligand_smiles"]
        neg_ligand_smiles = batch["neg_ligand_smiles"]
        neg_Y = batch["neg_Y"]

        protein_emb = self.encode_protein(protein_seqs)
        pos_ligand_emb = self.encode_ligand(pos_ligand_smiles)
        neg_ligand_emb = self.encode_ligand(neg_ligand_smiles)

        loss, temp, margin_score = self.clip_loss(protein_emb, pos_ligand_emb, neg_ligand_emb, neg_Y)

        pos_sim = (protein_emb * pos_ligand_emb).sum(dim=-1)
        neg_sim = (protein_emb.unsqueeze(1) * neg_ligand_emb).sum(dim=-1)
        val_acc = (pos_sim.unsqueeze(1) > neg_sim).all(dim=1).float().mean()

        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", val_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_temperature", temp, logger=True, sync_dist=True)
        self.log("val_margin_score", margin_score, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }