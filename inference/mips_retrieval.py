"""
Protein-Ligand检索系统 - 移除亲和力字段版
核心调整：
1. 完全移除所有亲和力相关逻辑（数据库、匹配、展示）
2. 保留核心的蛋白/小分子检索+相似度计算
3. 优化日志输出，聚焦检索结果本身
"""
import torch
import numpy as np
import warnings
from tqdm import tqdm
import faiss
warnings.filterwarnings("ignore")

# ==================== 模型定义（独立可运行） ====================
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SimpleProteinEncoder(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=128, hidden_dim=256, proj_dim=256, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs):
        aa_vocab = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
                    'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13,
                    'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19, 'X':20}

        max_len = max(len(seq) for seq in seqs)
        batch_emb = []
        for seq in seqs:
            seq_ids = [aa_vocab.get(c, 20) for c in seq[:max_len]]
            seq_ids += [20] * (max_len - len(seq_ids))
            batch_emb.append(seq_ids)

        x = torch.tensor(batch_emb).to(self.embedding.weight.device)
        x = self.embedding(x)
        x = self.dropout(x)

        lstm_out, _ = self.lstm(x)
        lstm_pool = lstm_out.mean(dim=1)
        proj_out = self.proj(lstm_pool)

        return F.normalize(proj_out, p=2, dim=-1)

class ChemBERTaEncoder(nn.Module):
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", proj_dim=256, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_list):
        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.bert.device)

        bert_out = self.bert(**inputs)
        cls_out = bert_out.last_hidden_state[:, 0, :]
        cls_out = self.dropout(cls_out)

        proj_out = self.proj(cls_out)

        return F.normalize(proj_out, p=2, dim=-1)

class CLIPStyleDualTower(nn.Module):
    def __init__(self, protein_embed_dim=128, proj_dim=256, init_temperature=0.2,
                 dropout=0.2, **kwargs):
        super().__init__()
        self.protein_encoder = SimpleProteinEncoder(
            embed_dim=protein_embed_dim,
            proj_dim=proj_dim,
            dropout=dropout
        )

        self.ligand_encoder = ChemBERTaEncoder(
            proj_dim=proj_dim,
            dropout=dropout
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
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        return model

# ==================== 检索器核心类（移除亲和力版） ====================
class ProteinLigandRetriever:
    def __init__(self, checkpoint_path, device="cuda:0", temperature_scale=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"检索器初始化设备：{self.device}")

        # 加载模型
        print(f"加载模型 checkpoint：{checkpoint_path}")
        self.model = CLIPStyleDualTower.load_from_checkpoint(
            checkpoint_path,
            protein_embed_dim=256,
            proj_dim=256,
            init_temperature=0.2,
            dropout=0.2
        ).to(self.device)

        # 评估模式
        self.model.eval()

        # 固定温度系数
        self.model.temperature.data = torch.tensor(temperature_scale).to(self.device)
        print(f"✅ 手动调整模型温度系数为：{self.model.temperature.item():.8f}")

        # 初始化索引
        self.protein_index = None
        self.ligand_index = None
        self.protein_id2seq = {}
        self.ligand_id2smiles = {}
        self.raw_dataset_samples = []

    def preprocess_protein(self, seq):
        """蛋白序列预处理"""
        aa_vocab = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
                    'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13,
                    'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19, 'X':20}
        max_len = 1024

        seq = seq.upper()
        seq = ''.join([c for c in seq if c in aa_vocab.keys()])
        seq = seq[:max_len]
        return seq

    def preprocess_ligand(self, smiles):
        """SMILES预处理"""
        if not smiles or len(smiles) < 1:
            return "C"
        return smiles.replace(' ', '').lower()

    def encode_protein_batch(self, protein_seqs):
        """批量编码蛋白"""
        processed_seqs = [self.preprocess_protein(seq) for seq in protein_seqs]
        with torch.no_grad():
            protein_emb = self.model.encode_protein(processed_seqs)
        return protein_emb.cpu().numpy()

    def encode_ligand_batch(self, ligand_smiles):
        """批量编码小分子"""
        processed_smiles = [self.preprocess_ligand(smi) for smi in ligand_smiles]
        with torch.no_grad():
            ligand_emb = self.model.encode_ligand(processed_smiles)
        return ligand_emb.cpu().numpy()

    def build_indexes_from_dataset(self, dataset, max_proteins=2000, max_ligands=8000):
        """构建索引（移除亲和力数据库）"""
        print("加载数据集：BALM/BALM-benchmark - BindingDB_filtered")
        dataset = dataset["train"]

        # 过滤无效样本
        def filter_invalid(sample):
            return (sample["Target"] and len(sample["Target"]) > 10 and
                    sample["Drug"] and len(sample["Drug"]) > 1)

        dataset = dataset.filter(filter_invalid)
        self.raw_dataset_samples = [s for s in dataset]
        print(f"过滤后数据集样本数：{len(self.raw_dataset_samples)}")

        # 构建预处理→原始序列映射
        protein_proc2raw = {}
        ligand_proc2raw = {}

        for raw_p in dataset["Target"]:
            proc_p = self.preprocess_protein(raw_p)
            if proc_p not in protein_proc2raw:
                protein_proc2raw[proc_p] = raw_p

        for raw_l in dataset["Drug"]:
            proc_l = self.preprocess_ligand(raw_l)
            if proc_l not in ligand_proc2raw:
                ligand_proc2raw[proc_l] = raw_l

        # 提取唯一序列
        unique_proteins = list(protein_proc2raw.keys())[:max_proteins]
        unique_ligands = list(ligand_proc2raw.keys())[:max_ligands]
        print(f"有效样本：蛋白{len(unique_proteins)}条（限制{max_proteins}），小分子{len(unique_ligands)}条（限制{max_ligands}）")

        # 批量编码蛋白
        batch_size = 32
        protein_embs = []
        for i in tqdm(range(0, len(unique_proteins), batch_size), desc="编码protein"):
            batch_seqs = unique_proteins[i:i + batch_size]
            batch_embs = self.encode_protein_batch(batch_seqs)
            protein_embs.append(batch_embs)
        protein_embs = np.concatenate(protein_embs, axis=0)

        # 批量编码小分子
        ligand_embs = []
        for i in tqdm(range(0, len(unique_ligands), batch_size), desc="编码ligand"):
            batch_smiles = unique_ligands[i:i + batch_size]
            batch_embs = self.encode_ligand_batch(batch_smiles)
            ligand_embs.append(batch_embs)
        ligand_embs = np.concatenate(ligand_embs, axis=0)

        # 构建ID映射
        self.protein_id2seq = {
            i: {
                "processed": seq,
                "raw": protein_proc2raw.get(seq, seq)
            } for i, seq in enumerate(unique_proteins)
        }
        self.ligand_id2smiles = {
            i: {
                "processed": smi,
                "raw": ligand_proc2raw.get(smi, smi)
            } for i, smi in enumerate(unique_ligands)
        }

        # 构建FAISS索引
        self.protein_index = faiss.IndexFlatIP(256)
        self.protein_index.add(protein_embs)
        print(f"protein FAISS索引构建完成：{len(unique_proteins)}个样本，维度256")

        self.ligand_index = faiss.IndexFlatIP(256)
        self.ligand_index.add(ligand_embs)
        print(f"ligand FAISS索引构建完成：{len(unique_ligands)}个样本，维度256")

    def retrieve_ligands(self, protein_seq, top_k=10):
        """检索给定蛋白的高相似度小分子（移除亲和力）"""
        # 编码查询蛋白
        query_emb = self.encode_protein_batch([protein_seq])[0].reshape(1, -1)

        # FAISS检索
        distances, indices = self.ligand_index.search(query_emb, top_k)

        # 归一化相似度到0~1
        max_sim = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        min_sim = np.min(distances[0]) if len(distances[0]) > 0 else 0.0
        norm_distances = (distances[0] - min_sim) / (max_sim - min_sim + 1e-8)

        # 解析结果
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.ligand_id2smiles):
                ligand_info = self.ligand_id2smiles[idx]
                results.append({
                    "smiles": ligand_info["raw"],
                    "smiles_processed": ligand_info["processed"],
                    "similarity": norm_distances[i]
                })

        return results

    def retrieve_proteins(self, ligand_smiles, top_k=10):
        """检索给定小分子的高相似度蛋白（移除亲和力）"""
        # 编码查询小分子
        query_emb = self.encode_ligand_batch([ligand_smiles])[0].reshape(1, -1)

        # FAISS检索
        distances, indices = self.protein_index.search(query_emb, top_k)

        # 归一化相似度到0~1
        max_sim = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        min_sim = np.min(distances[0]) if len(distances[0]) > 0 else 0.0
        norm_distances = (distances[0] - min_sim) / (max_sim - min_sim + 1e-8)

        # 解析结果
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.protein_id2seq):
                protein_info = self.protein_id2seq[idx]
                results.append({
                    "protein_seq": protein_info["raw"],
                    "protein_processed": protein_info["processed"],
                    "similarity": norm_distances[i]
                })

        return results

# ==================== 主测试程序（移除亲和力版） ====================
if __name__ == "__main__":
    # 1. 配置参数
    CHECKPOINT_PATH = "../model/checkpoints/clip_tower_best-v9.ckpt"
    DEVICE = "cuda:0"
    TEMPERATURE_SCALE = 0.5

    # 2. 初始化检索器
    retriever = ProteinLigandRetriever(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        temperature_scale=TEMPERATURE_SCALE
    )

    # 3. 加载数据集并构建索引
    try:
        from datasets import load_dataset
        ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")
        retriever.build_indexes_from_dataset(
            ds,
            max_proteins=2000,
            max_ligands=8000
        )
    except Exception as e:
        print(f"加载数据集失败：{e}")
        exit(1)

    # 4. 选择测试样本
    print("\n=== 从数据集选择测试样本 ===")
    test_protein = ""
    test_smiles = ""
    target_protein = ""
    target_smiles = ""

    if len(retriever.raw_dataset_samples) > 0:
        # 选择第一个样本
        real_sample = retriever.raw_dataset_samples[0]
        test_protein = real_sample["Target"]
        test_smiles = real_sample["Drug"]
        target_protein = test_protein
        target_smiles = test_smiles

        print(f"📌 测试蛋白（前60字符）：{test_protein[:60]}...")
        print(f"📌 测试SMILES：{test_smiles}")
    else:
        # 兜底用例
        test_protein = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRIL"
        test_smiles = "Cc1ccc(CNS(=O)(=O)c2ccc(S(N)(=O)=O)s2)cc1"
        target_protein = test_protein
        target_smiles = test_smiles
        print("⚠️  数据集为空，使用兜底测试用例")

    # 5. 测试1：蛋白→小分子推荐
    print("\n=== 测试1：蛋白→小分子推荐 ===")
    print(f"查询蛋白：{test_protein[:60]}...")
    ligand_results = retriever.retrieve_ligands(test_protein, top_k=10)
    for i, res in enumerate(ligand_results):
        smiles_display = res['smiles'][:50] + "..." if len(res['smiles']) > 50 else res['smiles']
        print(f"Top{i+1}：SMILES={smiles_display} | 相似度={res['similarity']:.4f}")

    # 6. 测试2：小分子→蛋白推荐
    print("\n=== 测试2：小分子→蛋白推荐 ===")
    print(f"查询SMILES：{test_smiles}")
    protein_results = retriever.retrieve_proteins(test_smiles, top_k=10)
    for i, res in enumerate(protein_results):
        seq_display = res['protein_seq'][:60] + "..." if len(res['protein_seq']) > 60 else res['protein_seq']
        print(f"Top{i+1}：蛋白序列={seq_display} | 相似度={res['similarity']:.4f}")

    # 7. 测试3：数据集外蛋白（胰岛素）
    insulin_protein = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    print("\n=== 测试4：数据集外推荐（胰岛素） ===")
    print(f"查询胰岛素蛋白：{insulin_protein[:60]}...")
    insulin_results = retriever.retrieve_ligands(insulin_protein, top_k=3)
    max_sim = max([r['similarity'] for r in insulin_results] + [1e-8])
    for i, res in enumerate(insulin_results):
        smiles_display = res['smiles'][:50] + "..." if len(res['smiles']) > 50 else res['smiles']
        norm_sim = res['similarity'] / max_sim if max_sim > 1e-8 else 0.0
        print(f"Top{i+1}：SMILES={smiles_display} | 相似度={res['similarity']:.4f} | 归一化相似度={norm_sim:.4f}")

    # 8. 测试6：验证目标结合对检索
    print("\n=== 测试6：验证目标结合对检索 ===")
    target_smiles_norm = target_smiles.replace(' ', '').lower()
    retrieval_results = retriever.retrieve_ligands(target_protein, top_k=10)

    found = False
    for i, res in enumerate(retrieval_results):
        res_smiles_norm = res['smiles'].replace(' ', '').lower()
        if res_smiles_norm == target_smiles_norm or res_smiles_norm[:50] == target_smiles_norm[:50]:
            found = True
            print(f"✅ 在Top{i+1}找到目标小分子！")
            print(f"   SMILES：{res['smiles'][:80]}...")
            print(f"   相似度：{res['similarity']:.4f}")
            break

    if not found:
        print(f"❌ 未在Top10找到目标小分子，Top10结果：")
        for i, res in enumerate(retrieval_results):
            smiles_display = res['smiles'][:50] + "..." if len(res['smiles']) > 50 else res['smiles']
            print(f"Top{i+1}：SMILES={smiles_display} | 相似度={res['similarity']:.4f}")

    print("\n=== 所有测试完成 ===")
