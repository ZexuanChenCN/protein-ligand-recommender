"""基于ChemBERTa的小分子Embedding提取模块，适配双塔模型的小分子塔"""
import torch
import torch.nn as nn
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel
from .utils import batch_clean_smiles


class ChemBERTaEmbeddingExtractor(nn.Module):

    def __init__(
            self,
            model_name: str = "DeepChem/ChemBERTa-77M-MLM",
            hidden_dim: int = 384,  # ChemBERTa-77M的隐藏维度
            output_dim: int = 2560,
            freeze_encoder: bool = True,
            local_files_only: bool = False
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ChemBERTa初始化设备：{self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                cache_dir="./models/cache"
            )
        except Exception as e:
            print(f"加载Tokenizer失败，尝试下载：{e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models/cache")

        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                cache_dir="./models/cache"
            ).to(self.device)
        except Exception as e:
            print(f"加载Encoder失败，尝试下载：{e}")
            self.encoder = AutoModel.from_pretrained(model_name, cache_dir="./models/cache").to(self.device)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.projection = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim).to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        cleaned_smiles = batch_clean_smiles(smiles_list)
        if not cleaned_smiles:
            return torch.zeros((len(smiles_list), self.projection.out_features), device=self.device)

        encoded_input = self.tokenizer(
            cleaned_smiles,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.encoder(**encoded_input)
        pooled_emb = self.mean_pooling(model_output, encoded_input["attention_mask"])
        mol_emb = self.projection(pooled_emb)
        mol_emb = self.layer_norm(self.relu(mol_emb))
        return mol_emb

    def to(self, device: Union[str, torch.device]) -> "ChemBERTaEmbeddingExtractor":
        self.device = torch.device(device)
        self.encoder = self.encoder.to(device)
        self.projection = self.projection.to(device)
        self.layer_norm = self.layer_norm.to(device)
        return self