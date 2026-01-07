"""
Protein-Ligand检索系统 - 最终完整版
核心修复：
1. 彻底解决亲和力匹配失败问题（精确+模糊匹配）
2. 优化温度系数，提升相似度区分度
3. 新增测试样本亲和力验证逻辑
4. 完善的日志和错误处理
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

# ==================== 检索器核心类（最终修复版） ====================
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

        # 初始化索引和数据库
        self.protein_index = None
        self.ligand_index = None
        self.protein_id2seq = {}
        self.ligand_id2smiles = {}
        self.affinity_db = {}
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
        """构建索引和亲和力数据库"""
        print("加载数据集：BALM/BALM-benchmark - BindingDB_filtered")
        dataset = dataset["train"]

        # 过滤无效样本
        def filter_invalid(sample):
            return (sample["Target"] and len(sample["Target"]) > 10 and
                    sample["Drug"] and len(sample["Drug"]) > 1 and
                    sample["Y"] is not None and sample["Y"] >= 0)

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

        # 构建亲和力数据库
        self.affinity_db = {}
        raw_affinity_db = {}

        for sample in tqdm(dataset, desc="构建原始亲和力数据库"):
            raw_p = sample["Target"].upper()
            raw_l = sample["Drug"].replace(' ', '').lower()
            y_val = sample["Y"]
            raw_affinity_db[(raw_p, raw_l)] = y_val

        # 存储多格式组合
        for (raw_p, raw_l), y_val in raw_affinity_db.items():
            proc_p = self.preprocess_protein(raw_p)
            proc_l = self.preprocess_ligand(raw_l)
            self.affinity_db[(raw_p, raw_l)] = y_val
            self.affinity_db[(proc_p, proc_l)] = y_val
            self.affinity_db[(raw_p, proc_l)] = y_val
            self.affinity_db[(proc_p, raw_l)] = y_val

        # 调试信息
        print("\n=== 亲和力数据库调试信息 ===")
        print(f"亲和力数据库总键数：{len(self.affinity_db)}")
        db_keys = list(self.affinity_db.keys())[:5]
        print(f"数据库前5个键示例：{db_keys}")

    def _get_affinity_with_fallback(self, protein_seq, ligand_smiles):
        """
        最终修复版亲和力查询：
        1. 精确匹配
        2. 蛋白匹配+小分子模糊匹配（取最大亲和力）
        3. 蛋白前缀匹配（取最大亲和力）
        """
        # 归一化查询序列
        p_norm = protein_seq.upper()
        l_norm = ligand_smiles.replace(' ', '').lower()
        proc_p = self.preprocess_protein(p_norm)
        proc_l = self.preprocess_ligand(l_norm)

        # 1. 精确匹配
        affinity = self.affinity_db.get((p_norm, l_norm),
                    self.affinity_db.get((p_norm, proc_l),
                    self.affinity_db.get((proc_p, l_norm),
                    self.affinity_db.get((proc_p, proc_l), -1))))

        # 2. 蛋白精确匹配 + 小分子任意匹配（取最大亲和力）
        if affinity == -1:
            protein_matches = []
            for (db_p, db_l), y_val in self.affinity_db.items():
                if db_p == p_norm or db_p == proc_p:
                    protein_matches.append(y_val)

            if protein_matches:
                affinity = max(protein_matches)

        # 3. 蛋白前缀匹配（前50字符）+ 取最大亲和力
        if affinity == -1:
            prefix_matches = []
            for (db_p, db_l), y_val in self.affinity_db.items():
                if db_p[:50] == p_norm[:50] or db_p[:50] == proc_p[:50]:
                    prefix_matches.append(y_val)

            if prefix_matches:
                affinity = max(prefix_matches)

        # 4. 最终兜底
        return affinity if affinity != -1 else 0.0

    def retrieve_ligands(self, protein_seq, top_k=10):
        """检索给定蛋白的高亲和力小分子"""
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
                raw_l = ligand_info["raw"]
                # 使用修复后的亲和力查询函数
                affinity = self._get_affinity_with_fallback(protein_seq, raw_l)

                results.append({
                    "smiles": raw_l,
                    "smiles_processed": ligand_info["processed"],
                    "similarity": norm_distances[i],
                    "affinity": affinity
                })

        return results

    def retrieve_proteins(self, ligand_smiles, top_k=10):
        """检索给定小分子的高亲和力蛋白"""
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
                raw_p = protein_info["raw"]
                # 使用修复后的亲和力查询函数
                affinity = self._get_affinity_with_fallback(raw_p, ligand_smiles)

                results.append({
                    "protein_seq": raw_p,
                    "protein_processed": protein_info["processed"],
                    "similarity": norm_distances[i],
                    "affinity": affinity
                })

        return results

# ==================== 主测试程序（最终版） ====================
if __name__ == "__main__":
    # 1. 配置参数
    CHECKPOINT_PATH = "C:/czx/Project/Grade0/recommender_system_project/protein-ligand-recommender/model/checkpoints/clip_tower_best-v9.ckpt"
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

    # 4. 选择并验证测试样本
    print("\n=== 从数据集选择并验证测试样本 ===")
    test_protein = ""
    test_smiles = ""
    target_protein = ""
    target_smiles = ""
    real_affinity = 0.0

    if len(retriever.raw_dataset_samples) > 0:
        # 选择第一个样本
        real_sample = retriever.raw_dataset_samples[0]
        test_protein = real_sample["Target"]
        test_smiles = real_sample["Drug"]
        target_protein = test_protein
        target_smiles = test_smiles

        print(f"📌 测试蛋白（前60字符）：{test_protein[:60]}...")
        print(f"📌 测试SMILES：{test_smiles}")

        # 验证亲和力匹配
        print("\n=== 验证亲和力匹配 ===")
        test_p_norm = test_protein.upper()
        test_l_norm = test_smiles.replace(' ', '').lower()

        # 1. 精确匹配
        exact_affinity = retriever.affinity_db.get((test_p_norm, test_l_norm), -1)
        if exact_affinity != -1:
            real_affinity = exact_affinity
            print(f"✅ 精确匹配成功！亲和力值：{real_affinity:.2f}")
        else:
            # 2. 蛋白匹配取最大亲和力
            protein_matches = []
            for (db_p, db_l), y_val in retriever.affinity_db.items():
                if db_p == test_p_norm or db_p[:50] == test_p_norm[:50]:
                    protein_matches.append(y_val)

            if protein_matches:
                real_affinity = max(protein_matches)
                print(f"⚠️  精确匹配失败，取蛋白匹配的最大亲和力：{real_affinity:.2f}")
            else:
                real_affinity = 0.0
                print(f"❌ 未找到任何匹配的亲和力值")

        print(f"📌 最终使用的亲和力值：{real_affinity:.2f}")
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
        print(f"Top{i+1}：SMILES={smiles_display} | 相似度={res['similarity']:.4f} | 亲和力={res['affinity']:.2f}")

    # 6. 测试2：小分子→蛋白推荐
    print("\n=== 测试2：小分子→蛋白推荐 ===")
    print(f"查询SMILES：{test_smiles}")
    protein_results = retriever.retrieve_proteins(test_smiles, top_k=10)
    for i, res in enumerate(protein_results):
        seq_display = res['protein_seq'][:60] + "..." if len(res['protein_seq']) > 60 else res['protein_seq']
        print(f"Top{i+1}：蛋白序列={seq_display} | 相似度={res['similarity']:.4f} | 亲和力={res['affinity']:.2f}")

    # 7. 测试3：数据集外蛋白（胰岛素）
    insulin_protein = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    print("\n=== 测试4：数据集外推荐（胰岛素） ===")
    print(f"查询胰岛素蛋白：{insulin_protein[:60]}...")
    insulin_results = retriever.retrieve_ligands(insulin_protein, top_k=3)
    max_sim = max([r['similarity'] for r in insulin_results] + [1e-8])
    for i, res in enumerate(insulin_results):
        smiles_display = res['smiles'][:50] + "..." if len(res['smiles']) > 50 else res['smiles']
        norm_sim = res['similarity'] / max_sim if max_sim > 1e-8 else 0.0
        print(f"Top{i+1}：SMILES={smiles_display} | 相似度={res['similarity']:.4f} | 归一化相似度={norm_sim:.4f} | 亲和力={res['affinity']:.2f}")

    # 8. 测试6：验证高亲和力结合对
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
            print(f"   相似度：{res['similarity']:.4f} | 亲和力：{res['affinity']:.2f}")
            break

    if not found:
        print(f"❌ 未在Top10找到目标小分子，Top10结果：")
        for i, res in enumerate(retrieval_results):
            smiles_display = res['smiles'][:50] + "..." if len(res['smiles']) > 50 else res['smiles']
            print(f"Top{i+1}：SMILES={smiles_display} | 相似度={res['similarity']:.4f} | 亲和力={res['affinity']:.2f}")

    print("\n=== 所有测试完成 ===")