"""
Protein-Ligand检索系统 - 性能优化版
核心提速点：
1. 蛋白编码向量化（替换手动list循环）
2. 开启推理模式+混合精度
3. 批量处理优化
4. 预编译模型
"""
import torch
import numpy as np
import warnings
from tqdm import tqdm
import faiss
import os
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 开启混合精度推理（和训练时一致）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

class SaProtProteinEncoder(nn.Module):
    """SaProt蛋白编码器（性能优化版）"""
    def __init__(self, saprot_model_dir, proj_dim=256, dropout=0.2, freeze_backbone=True):
        super().__init__()
        self.saprot_model_dir = Path(saprot_model_dir).resolve()
        print(f"📌 加载本地SaProt模型结构：{self.saprot_model_dir}")

        config_path = self.saprot_model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"未找到SaProt配置文件：{config_path}")

        self.config = AutoConfig.from_pretrained(
            str(config_path),
            local_files_only=True,
            ignore_mismatched_sizes=True
        )

        self.bert = AutoModel.from_pretrained(
            str(self.saprot_model_dir),
            config=self.config,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.aa_vocab = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
                         'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13,
                         'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19, 'X':20}
        self.vocab_size = len(self.aa_vocab)
        self.max_len = 1024

    def tokenize_batch_vectorized(self, seqs):
        """向量化批量tokenize"""
        seqs = [seq.upper()[:self.max_len] for seq in seqs]
        seq_ids = []
        for seq in seqs:
            ids = [self.aa_vocab.get(c, 20) for c in seq if c in self.aa_vocab]
            seq_ids.append(ids)

        max_batch_len = min(max(len(ids) for ids in seq_ids), self.max_len)
        input_ids = torch.zeros((len(seqs), max_batch_len), dtype=torch.long)
        attention_masks = torch.zeros((len(seqs), max_batch_len), dtype=torch.long)

        for i, ids in enumerate(seq_ids):
            valid_len = min(len(ids), max_batch_len)
            input_ids[i, :valid_len] = torch.tensor(ids[:valid_len])
            attention_masks[i, :valid_len] = 1

        return input_ids, attention_masks

    def forward(self, seqs):
        """前向传播（"""
        input_ids, attention_masks = self.tokenize_batch_vectorized(seqs)

        input_ids = input_ids.to(self.bert.device)
        attention_masks = attention_masks.to(self.bert.device)

        with torch.inference_mode():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_masks)
            cls_out = bert_out.last_hidden_state[:, 0, :]
            cls_out = self.dropout(cls_out)
            proj_out = self.proj(cls_out)

        return F.normalize(proj_out, p=2, dim=-1)

class ChemBERTaEncoder(nn.Module):
    """ChemBERTa编码器"""
    def __init__(self, chemberta_model_name, proj_dim=256, dropout=0.2, freeze_backbone=True):
        super().__init__()
        print(f"📌 加载ChemBERTa模型：{chemberta_model_name}")

        self.bert = AutoModel.from_pretrained(
            chemberta_model_name,
            local_files_only=False,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            chemberta_model_name,
            local_files_only=False,
            trust_remote_code=True
        )

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_list):
        """前向传播"""
        with torch.inference_mode():
            inputs = self.tokenizer(
                smiles_list, padding=True, truncation=True, max_length=128,
                return_tensors="pt", return_attention_mask=True
            ).to(self.bert.device)

            bert_out = self.bert(**inputs)
            cls_out = bert_out.last_hidden_state[:, 0, :]
            cls_out = self.dropout(cls_out)
            proj_out = self.proj(cls_out)

        return F.normalize(proj_out, p=2, dim=-1)

class CLIPStyleDualTower(nn.Module):
    """双塔模型"""
    def __init__(self, saprot_model_dir, chemberta_model_name,
                 proj_dim=256, init_temperature=0.2, dropout=0.2,
                 freeze_saprot=True, freeze_chemberta=True):
        super().__init__()
        self.protein_encoder = SaProtProteinEncoder(
            saprot_model_dir=saprot_model_dir,
            proj_dim=proj_dim,
            dropout=dropout,
            freeze_backbone=freeze_saprot
        )
        self.ligand_encoder = ChemBERTaEncoder(
            chemberta_model_name=chemberta_model_name,
            proj_dim=proj_dim,
            dropout=dropout,
            freeze_backbone=freeze_chemberta
        )

        self.temperature = torch.nn.Parameter(torch.tensor(init_temperature))
        self.proj_dim = proj_dim

    def encode_protein(self, protein_seqs):
        return self.protein_encoder(protein_seqs)

    def encode_ligand(self, ligand_smiles):
        return self.ligand_encoder(ligand_smiles)

    def forward(self, protein_seqs, ligand_smiles, neg_ligand_smiles=None):
        protein_embs = self.encode_protein(protein_seqs)
        ligand_embs = self.encode_ligand(ligand_smiles)
        sim = torch.matmul(protein_embs, ligand_embs.t()) / self.temperature

        if neg_ligand_smiles is not None:
            neg_ligand_embs = self.encode_ligand(neg_ligand_smiles)
            neg_sim = torch.matmul(protein_embs, neg_ligand_embs.t()) / self.temperature
            return sim, neg_sim
        return sim

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"📌 从ckpt加载训练参数：{checkpoint_path}")

        hparams = checkpoint.get("hyper_parameters", {})
        load_kwargs = {
            "saprot_model_dir": kwargs.get("saprot_model_dir", hparams.get("saprot_model_dir")),
            "chemberta_model_name": kwargs.get("chemberta_model_name", hparams.get("chemberta_model_name")),
            "proj_dim": kwargs.get("proj_dim", hparams.get("proj_dim", 256)),
            "init_temperature": kwargs.get("init_temperature", hparams.get("init_temperature", 0.2)),
            "dropout": kwargs.get("dropout", hparams.get("dropout", 0.2)),
            "freeze_saprot": kwargs.get("freeze_saprot", hparams.get("freeze_saprot", True)),
            "freeze_chemberta": kwargs.get("freeze_chemberta", hparams.get("freeze_chemberta", True))
        }

        model = cls(**load_kwargs)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["state_dict"],
            strict=False
        )
        print(f"⚠️  权重加载统计 - 缺失keys: {len(missing_keys)} | 额外keys: {len(unexpected_keys)}")

        # 模型编译（PyTorch 2.0+，大幅提速）
        if torch.__version__ >= "2.0.0":
            model = torch.compile(model)

        return model

class ProteinLigandRetriever:
    def __init__(self, checkpoint_path, saprot_model_dir, chemberta_model_name,
                 device="cuda:0", temperature_scale=0.5, batch_size=64):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"检索器初始化设备：{self.device}")

        # 加载模型
        self.model = CLIPStyleDualTower.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            saprot_model_dir=saprot_model_dir,
            chemberta_model_name=chemberta_model_name,
            freeze_saprot=True,
            freeze_chemberta=True
        ).to(self.device)

        # 极致推理模式
        self.model.eval()
        self.model = self.model.half()  # 混合精度推理（FP16）

        self.model.temperature.data = torch.tensor(temperature_scale).to(self.device).half()
        print(f"✅ 温度系数设置为：{self.model.temperature.item():.8f}")

        # 批量大小优化
        self.batch_size = batch_size
        self.protein_index = None
        self.ligand_index = None
        self.protein_id2seq = {}
        self.ligand_id2smiles = {}
        self.raw_dataset_samples = []

    def preprocess_protein(self, seq):
        """快速预处理"""
        aa_vocab = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
                    'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13,
                    'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19, 'X':20}
        max_len = 1024
        seq = seq.upper()
        seq = ''.join([c for c in seq if c in aa_vocab.keys()])
        return seq[:max_len]

    def preprocess_ligand(self, smiles):
        if not smiles or len(smiles) < 1:
            return "C"
        return smiles.replace(' ', '').lower()

    def encode_protein_batch(self, protein_seqs):
        """批量编码"""
        processed_seqs = [self.preprocess_protein(seq) for seq in protein_seqs]
        with torch.no_grad():
            # 混合精度推理
            protein_emb = self.model.encode_protein(processed_seqs).float()
        return protein_emb.cpu().numpy()

    def encode_ligand_batch(self, ligand_smiles):
        """批量编码"""
        processed_smiles = [self.preprocess_ligand(smi) for smi in ligand_smiles]
        with torch.no_grad():
            ligand_emb = self.model.encode_ligand(processed_smiles).float()
        return ligand_emb.cpu().numpy()

    def build_indexes_from_dataset(self, dataset, max_proteins=2000, max_ligands=8000):
        print("📥 加载数据集：BALM/BALM-benchmark - BindingDB_filtered")
        dataset = dataset["train"]

        def filter_invalid(sample):
            return len(sample["Target"])>10 and len(sample["Drug"])>1
        dataset = dataset.filter(filter_invalid)
        self.raw_dataset_samples = [s for s in dataset]
        print(f"✅ 过滤后有效样本数：{len(self.raw_dataset_samples)}")

        protein_proc2raw = {self.preprocess_protein(p):p for p in dataset["Target"]}
        ligand_proc2raw = {self.preprocess_ligand(l):l for l in dataset["Drug"]}

        unique_proteins = list(protein_proc2raw.keys())[:max_proteins]
        unique_ligands = list(ligand_proc2raw.keys())[:max_ligands]
        print(f"📊 构建索引 - 蛋白数：{len(unique_proteins)} | 小分子数：{len(unique_ligands)}")

        protein_embs = []
        for i in tqdm(range(0, len(unique_proteins), self.batch_size), desc="编码蛋白序列"):
            batch_seqs = unique_proteins[i:i+self.batch_size]
            batch_embs = self.encode_protein_batch(batch_seqs)
            protein_embs.append(batch_embs)
        protein_embs = np.concatenate(protein_embs, axis=0)

        ligand_embs = []
        for i in tqdm(range(0, len(unique_ligands), self.batch_size), desc="编码小分子SMILES"):
            batch_smiles = unique_ligands[i:i+self.batch_size]
            batch_embs = self.encode_ligand_batch(batch_smiles)
            ligand_embs.append(batch_embs)
        ligand_embs = np.concatenate(ligand_embs, axis=0)

        self.protein_id2seq = {i:{"processed":s,"raw":protein_proc2raw[s]} for i,s in enumerate(unique_proteins)}
        self.ligand_id2smiles = {i:{"processed":s,"raw":ligand_proc2raw[s]} for i,s in enumerate(unique_ligands)}

        self.protein_index = faiss.IndexFlatIP(256)
        self.protein_index.add(protein_embs)
        self.ligand_index = faiss.IndexFlatIP(256)
        self.ligand_index.add(ligand_embs)

        print(f"✅ FAISS索引构建完成 - 蛋白索引数：{self.protein_index.ntotal} | 小分子索引数：{self.ligand_index.ntotal}")

    def retrieve_ligands(self, protein_seq, top_k=10):
        query_emb = self.encode_protein_batch([protein_seq])[0].reshape(1, -1)
        distances, indices = self.ligand_index.search(query_emb, top_k)

        max_sim = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        min_sim = np.min(distances[0]) if len(distances[0]) > 0 else 0.0
        norm_distances = (distances[0] - min_sim) / (max_sim - min_sim + 1e-8)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.ligand_id2smiles):
                info = self.ligand_id2smiles[idx]
                results.append({
                    "smiles": info["raw"],
                    "smiles_processed": info["processed"],
                    "similarity": float(norm_distances[i])
                })
        return results

    def retrieve_proteins(self, ligand_smiles, top_k=10):
        query_emb = self.encode_ligand_batch([ligand_smiles])[0].reshape(1, -1)
        distances, indices = self.protein_index.search(query_emb, top_k)

        max_sim = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        min_sim = np.min(distances[0]) if len(distances[0]) > 0 else 0.0
        norm_distances = (distances[0] - min_sim) / (max_sim - min_sim + 1e-8)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.protein_id2seq):
                info = self.protein_id2seq[idx]
                results.append({
                    "protein_seq": info["raw"],
                    "protein_processed": info["processed"],
                    "similarity": float(norm_distances[i])
                })
        return results


if __name__ == "__main__":
    CHECKPOINT_PATH = "../model/checkpoints/saprot_clip_tower_best.ckpt"
    SAPROT_MODEL_DIR = "../models/SaProt_1.3B_AFDB_OMG_NCBI"
    CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
    TEMPERATURE_SCALE = 0.5
    DEVICE = "cuda:0"
    BATCH_SIZE = 64

    # 路径校验
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 错误：ckpt文件不存在 - {CHECKPOINT_PATH}")
        sys.exit(1)
    if not os.path.exists(SAPROT_MODEL_DIR):
        print(f"❌ 错误：SaProt模型目录不存在 - {SAPROT_MODEL_DIR}")
        sys.exit(1)

    try:
        retriever = ProteinLigandRetriever(
            checkpoint_path=CHECKPOINT_PATH,
            saprot_model_dir=SAPROT_MODEL_DIR,
            chemberta_model_name=CHEMBERTA_MODEL_NAME,
            device=DEVICE,
            temperature_scale=TEMPERATURE_SCALE,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"❌ 检索器初始化失败：{str(e)}")
        sys.exit(1)

    try:
        from datasets import load_dataset
        ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")
        retriever.build_indexes_from_dataset(ds, max_proteins=2000, max_ligands=8000)
    except Exception as e:
        print(f"❌ 数据集加载/索引构建失败：{str(e)}")
        sys.exit(1)

    print("\n" + "="*50)
    print("📝 开始测试检索功能")
    print("="*50)

    if len(retriever.raw_dataset_samples) > 0:
        test_protein = retriever.raw_dataset_samples[0]["Target"]
        test_smiles = retriever.raw_dataset_samples[0]["Drug"]
    else:
        test_protein = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRIL"
        test_smiles = "Cc1ccc(CNS(=O)(=O)c2ccc(S(N)(=O)=O)s2)cc1"

    print(f"\n🔍 测试1：蛋白序列 → 小分子检索")
    print(f"查询蛋白（前50字符）：{test_protein[:50]}...")
    ligand_results = retriever.retrieve_ligands(test_protein, top_k=5)
    for i, res in enumerate(ligand_results):
        print(f"  Top{i+1} - SMILES：{res['smiles'][:50]} | 相似度：{res['similarity']:.4f}")


    print(f"\n🔍 测试2：小分子SMILES → 蛋白检索")
    print(f"查询SMILES：{test_smiles}")
    protein_results = retriever.retrieve_proteins(test_smiles, top_k=5)
    for i, res in enumerate(protein_results):
        print(f"  Top{i+1} - 蛋白（前50字符）：{res['protein_seq'][:50]}... | 相似度：{res['similarity']:.4f}")

    print("\n✅ 所有测试完成！")