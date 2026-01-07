"""
data/dataloader.py - 最终可运行版（修复切片语法+全边界容错）
"""
import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
import warnings
warnings.filterwarnings("ignore")

class ContrastivePLDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: Dataset,
            pos_threshold: float = 6.0,
            margin: float = 0.5,
            num_neg_samples: int = 8,
            filter_min_samples: int = 5  # 柔性过滤：仅过滤样本数<5的蛋白
    ):
        super().__init__()
        self.dataset = dataset
        self.pos_threshold = pos_threshold
        self.margin = margin
        self.num_neg_samples = num_neg_samples
        self.filter_min_samples = filter_min_samples

        # 提取核心字段
        self.all_Y = dataset["Y"]
        self.all_targets = dataset["Target_ID"]
        self.all_proteins = dataset["Target"]
        self.all_ligands = dataset["Drug"]

        # 构建Target到样本索引的映射
        self.target2indices: Dict[str, List[int]] = {}
        for idx, target_id in enumerate(self.all_targets):
            if target_id not in self.target2indices:
                self.target2indices[target_id] = []
            self.target2indices[target_id].append(idx)

        # 柔性过滤：保留样本数≥filter_min_samples的蛋白（兜底逻辑）
        self.valid_targets = [
            tid for tid in self.target2indices
            if len(self.target2indices[tid]) >= self.filter_min_samples
        ]
        # 兜底：如果有效蛋白太少，放宽到3个样本
        if len(self.valid_targets) < 100:
            self.valid_targets = [
                tid for tid in self.target2indices
                if len(self.target2indices[tid]) >= 3
            ]
        if len(self.valid_targets) == 0:
            raise ValueError(f"无有效蛋白样本（至少需要{self.filter_min_samples}个样本/蛋白）")

        # 分层负样本池
        self.core_targets = [tid for tid in self.valid_targets if len(self.target2indices[tid]) >= 5]
        self.minor_targets = [tid for tid in self.valid_targets if len(self.target2indices[tid]) < 5]

        # 全局负样本池（按亲和力分层）
        self.high_affinity_neg = [idx for idx in range(len(dataset)) if 3 <= self.all_Y[idx] < self.pos_threshold]
        self.low_affinity_neg = [idx for idx in range(len(dataset)) if self.all_Y[idx] < 3]
        self.global_cross_target_neg = self.high_affinity_neg + self.low_affinity_neg
        if len(self.global_cross_target_neg) == 0:
            self.global_cross_target_neg = list(range(len(dataset)))

    def _sample_stratified_neg(self, target_id, current_idx):
        """分层负样本采样：核心靶点同蛋白为主，小众靶点跨蛋白为主（彻底修复语法错误）"""
        sample_indices = self.target2indices[target_id]
        # 1. 同蛋白负样本（容错：无同蛋白负样本则为空）
        same_neg = [i for i in sample_indices if i != current_idx and self.all_Y[i] < self.pos_threshold]

        # 2. 分层策略
        if target_id in self.core_targets:
            # 核心靶点：60%同蛋白 + 40%跨蛋白
            num_same = max(1, int(self.num_neg_samples * 0.6))
            num_cross = self.num_neg_samples - num_same
        else:
            # 小众靶点：20%同蛋白 + 80%跨蛋白
            num_same = max(1, int(self.num_neg_samples * 0.2))
            num_cross = self.num_neg_samples - num_same

        # 3. 采样同蛋白负样本（彻底修复语法错误+全容错）
        neg_indices = []
        if len(same_neg) > 0:
            if len(same_neg) >= num_same:
                # 足够样本：直接采样
                neg_indices.extend(random.sample(same_neg, k=num_same))
            else:
                # 样本不足：先复制再截断（修复核心语法错误）
                repeat_times = num_same // len(same_neg) + 1
                temp_list = same_neg * repeat_times
                neg_indices.extend(temp_list[:num_same])  # 先extend再切片，而非链式操作
        else:
            # 无同蛋白负样本：全部用跨蛋白负样本
            num_same = 0
            num_cross = self.num_neg_samples

        # 4. 采样跨蛋白负样本（优先高亲和力负样本，极端容错）
        cross_neg_candidates = [i for i in self.global_cross_target_neg if i not in neg_indices and i != current_idx]
        # 兜底1：候选为空则用所有样本
        if len(cross_neg_candidates) == 0:
            cross_neg_candidates = list(range(len(self.dataset)))
        # 兜底2：候选不足则重复填充
        while len(cross_neg_candidates) < num_cross:
            cross_neg_candidates.extend(cross_neg_candidates)
        # 采样跨蛋白负样本
        cross_neg = random.sample(cross_neg_candidates, k=num_cross)
        neg_indices.extend(cross_neg)

        # 5. 最终兜底：确保数量足够+去重
        neg_indices = list(dict.fromkeys(neg_indices))  # 去重
        while len(neg_indices) < self.num_neg_samples:
            neg_indices.append(neg_indices[-1] if neg_indices else current_idx)
        # 截断到目标数量
        neg_indices = neg_indices[:self.num_neg_samples]

        return neg_indices

    def _augment_smiles(self, smiles: str) -> str:
        """SMILES简单增强：随机替换氢原子标记"""
        if not smiles or len(smiles) < 3:
            return smiles
        # 训练时才增强
        if self.training and random.random() < 0.3:
            # 随机替换[H]为H
            smiles = smiles.replace('[H]', 'H')
            # 随机在碳后加H（简单增强）
            chars = list(smiles)
            aug_chars = []
            for c in chars:
                aug_chars.append(c)
                if c == 'C' and random.random() < 0.2:
                    aug_chars.append('H')
            return ''.join(aug_chars)
        return smiles

    def __getitem__(self, idx: int) -> Dict:
        target_id = self.valid_targets[idx]
        sample_indices = self.target2indices[target_id]

        # 正负样本筛选
        pos_candidates = [i for i in sample_indices if self.all_Y[i] >= self.pos_threshold]

        # 正样本兜底
        if not pos_candidates:
            pos_idx = max(sample_indices, key=lambda i: self.all_Y[i])
        else:
            pos_idx = random.choice(pos_candidates)
        pos_sample = self.dataset[pos_idx]
        pos_smiles = pos_sample["Drug"] if pos_sample["Drug"] else "C"
        # SMILES增强
        pos_smiles_aug = self._augment_smiles(pos_smiles)
        pos_aug_smiles = [pos_smiles, pos_smiles_aug]

        # 分层采样负样本（已修复所有错误）
        neg_indices = self._sample_stratified_neg(target_id, pos_idx)

        # 收集负样本SMILES（带增强）
        neg_smiles = []
        neg_Y_list = []
        for neg_idx in neg_indices:
            smi = self.dataset[neg_idx]["Drug"] if self.dataset[neg_idx]["Drug"] else "C"
            neg_smiles.append(self._augment_smiles(smi))
            neg_Y_list.append(self.all_Y[neg_idx])

        return {
            "protein_seqs": pos_sample["Target"],
            "pos_ligand_smiles": pos_smiles,
            "pos_aug_smiles": pos_aug_smiles,
            "neg_ligand_smiles": neg_smiles,
            "pos_Y": self.all_Y[pos_idx],
            "neg_Y": neg_Y_list,
            "target_id": target_id
        }

    def __len__(self) -> int:
        return len(self.valid_targets)

def get_dataloaders(
        batch_size: int = 4,
        num_workers: int = 0,
        pos_threshold: float = 6.0,
        margin: float = 0.5,
        num_neg_samples: int = 8,
        filter_min_samples: int = 5
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    print("加载BALM数据集...")
    ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")

    print(f"数据集可用划分：{list(ds.keys())}")

    # 容错逻辑
    if "validation" in ds:
        train_ds_raw = ds["train"]
        val_ds_raw = ds["validation"]
    elif "val" in ds:
        train_ds_raw = ds["train"]
        val_ds_raw = ds["val"]
    elif "test" in ds:
        train_ds_raw = ds["train"]
        val_ds_raw = ds["test"]
    else:
        train_ds_raw = ds["train"]
        unique_targets = list(set(train_ds_raw["Target_ID"]))
        random.seed(42)
        val_targets = random.sample(unique_targets, k=int(len(unique_targets)*0.1))
        val_ds_raw = train_ds_raw.filter(lambda x: x["Target_ID"] in val_targets)
        train_ds_raw = train_ds_raw.filter(lambda x: x["Target_ID"] not in val_targets)
        print(f"从train集拆分验证集：训练集{len(train_ds_raw)}条，验证集{len(val_ds_raw)}条")

    # 过滤无效样本
    def filter_invalid(sample):
        return (sample["Target"] and len(sample["Target"]) > 10 and
                sample["Drug"] and len(sample["Drug"]) > 1 and
                sample["Y"] is not None and sample["Y"] >= 0)

    train_ds_raw = train_ds_raw.filter(filter_invalid)
    val_ds_raw = val_ds_raw.filter(filter_invalid)

    # 构建数据集
    train_dataset = ContrastivePLDataset(
        train_ds_raw,
        pos_threshold=pos_threshold,
        margin=margin,
        num_neg_samples=num_neg_samples,
        filter_min_samples=filter_min_samples
    )
    val_dataset = ContrastivePLDataset(
        val_ds_raw,
        pos_threshold=pos_threshold,
        margin=margin,
        num_neg_samples=num_neg_samples,
        filter_min_samples=filter_min_samples
    )
    # 标记训练/验证模式（用于数据增强）
    train_dataset.training = True
    val_dataset.training = False

    # collate_fn
    def collate_fn(batch):
        batch_dict = {
            "protein_seqs": [item["protein_seqs"] for item in batch],
            "pos_ligand_smiles": [item["pos_ligand_smiles"] for item in batch],
            "pos_aug_smiles": [item["pos_aug_smiles"] for item in batch],
            "neg_ligand_smiles": [item["neg_ligand_smiles"] for item in batch],
            "pos_Y": torch.tensor([item["pos_Y"] for item in batch], dtype=torch.float32),
            "neg_Y": torch.tensor([item["neg_Y"] for item in batch], dtype=torch.float32),
            "target_id": [item["target_id"] for item in batch]
        }
        return batch_dict

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

    print(f"数据集加载完成：训练集{len(train_loader)}批次，验证集{len(val_loader)}批次")
    print(f"参与训练的蛋白数：{len(train_dataset.valid_targets)}（核心靶点：{len(train_dataset.core_targets)}）")
    return train_loader, val_loader