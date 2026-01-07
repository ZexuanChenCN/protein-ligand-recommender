"""基于ChemBERTa的小分子Embedding提取模块，适配双塔模型的小分子塔"""
import torch
import torch.nn as nn
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel
from .utils import batch_clean_smiles


class ChemBERTaEmbeddingExtractor(nn.Module):
    """封装ChemBERTa为可训练的小分子塔，输出固定维度Embedding"""

    def __init__(
            self,
            model_name: str = "DeepChem/ChemBERTa-77M-MLM",
            hidden_dim: int = 384,  # ChemBERTa-77M的隐藏维度
            output_dim: int = 2560,
            freeze_encoder: bool = True,
            local_files_only: bool = False
    ):
        super().__init__()

        # 1. 自动检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ChemBERTa初始化设备：{self.device}")

        # 2. 加载Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                cache_dir="./models/cache"
            )
        except Exception as e:
            print(f"加载Tokenizer失败，尝试下载：{e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models/cache")

        # 3. 加载Encoder
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                cache_dir="./models/cache"
            ).to(self.device)
        except Exception as e:
            print(f"加载Encoder失败，尝试下载：{e}")
            self.encoder = AutoModel.from_pretrained(model_name, cache_dir="./models/cache").to(self.device)

        # 冻结编码器
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()  # 特征提取模式

        # 4. 投影层+归一化层
        self.projection = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim).to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        """均值池化：融合token级Embedding为分子级Embedding"""
        token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """前向传播：输入SMILES列表，输出小分子Embedding"""
        # 1. SMILES预处理
        cleaned_smiles = batch_clean_smiles(smiles_list)
        if not cleaned_smiles:
            # 返回全零张量避免报错
            return torch.zeros((len(smiles_list), self.projection.out_features), device=self.device)

        # 2. Tokenize
        encoded_input = self.tokenizer(
            cleaned_smiles,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # 3. 生成Token级Embedding（关闭梯度）
        with torch.no_grad():
            model_output = self.encoder(**encoded_input)

        # 4. 均值池化
        pooled_emb = self.mean_pooling(model_output, encoded_input["attention_mask"])

        # 5. 维度统一+归一化
        mol_emb = self.projection(pooled_emb)
        mol_emb = self.layer_norm(self.relu(mol_emb))

        return mol_emb

    # 简化设备迁移逻辑
    def to(self, device: Union[str, torch.device]) -> "ChemBERTaEmbeddingExtractor":
        self.device = torch.device(device)
        self.encoder = self.encoder.to(device)
        self.projection = self.projection.to(device)
        self.layer_norm = self.layer_norm.to(device)
        return self