"""小分子处理辅助函数：SMILES清洗（无rdkit依赖）"""
from typing import List, Optional

def clean_smiles(smiles: str) -> Optional[str]:
    """清洗SMILES：仅过滤空值和无效字符，不使用rdkit"""
    if not smiles:
        return None
    valid_chars = set("CNOSPFBrlatc()[]=+-\\/@%.0123456789")
    cleaned = ''.join([c for c in smiles if c in valid_chars])
    return cleaned if len(cleaned) > 1 else None

def batch_clean_smiles(smiles_list: List[str]) -> List[str]:
    """批量清洗SMILES列表，过滤无效值"""
    cleaned = []
    for smi in smiles_list:
        cleaned_smi = clean_smiles(smi)
        if cleaned_smi:
            cleaned.append(cleaned_smi)
    if not cleaned:
        cleaned = ["C"]
    return cleaned