import torch
import numpy as np  # 补充缺失的导入
from typing import Optional, List, Union
from transformers import EsmTokenizer
from model.saprot.base import SaprotBaseModel


class SaProtEmbeddingExtractor:
    """SaProt蛋白质嵌入提取器，用于从蛋白质序列（含结构信息）中提取特征嵌入"""

    def __init__(
            self,
            model_path: str,
            device: Optional[str] = None,
            use_half_precision: bool = False
    ):
        """
        初始化嵌入提取器

        Args:
            model_path: 模型权重目录路径（如SaProt_650M_AF2）
            device: 运行设备（"cuda"或"cpu"），默认自动检测
            use_half_precision: 是否使用半精度浮点数加速计算
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = use_half_precision

        # 加载模型和分词器
        self._load_model()
        self._load_tokenizer()

        # 模型模式设置
        self.model.eval()
        if self.use_half_precision:
            self.model.half()

    def _load_model(self) -> None:
        """加载SaProt基础模型"""
        config = {
            "task": "base",
            "config_path": self.model_path,
            "load_pretrained": True,
        }
        self.model = SaprotBaseModel(**config)
        self.model.to(self.device)

    def _load_tokenizer(self) -> None:
        """加载ESMFold分词器"""
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_path)

    def process_sequence(
            self,
            sequence: str
    ) -> dict:
        """
        处理输入序列，转换为模型可接受的格式

        Args:
            sequence: 蛋白质序列（可包含#标记低plDDT区域，如"M#EvVpQpL#VyQdYaKv"）

        Returns:
            模型输入字典（含input_ids、attention_mask等）
        """
        inputs = self.tokenizer(sequence, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_embedding(
            self,
            sequence: Union[str, List[str]],
            reduction: Optional[str] = "mean",
            return_numpy: bool = True
    ) -> Union[List[torch.Tensor], List[np.ndarray]]:
        """
        提取蛋白质序列的嵌入特征

        Args:
            sequence: 单个蛋白质序列或序列列表
            reduction: 嵌入聚合方式（None表示返回完整序列嵌入，"mean"表示平均池化）
            return_numpy: 是否转换为numpy数组

        Returns:
            嵌入特征列表，每个元素对应一个输入序列的嵌入
        """
        # 处理批量输入
        if isinstance(sequence, str):
            sequences = [sequence]
        else:
            sequences = sequence

        embeddings = []
        with torch.no_grad():
            for seq in sequences:
                # 序列预处理
                inputs = self.process_sequence(seq)

                # 提取隐藏状态
                hidden_states = self.model.get_hidden_states(
                    inputs,
                    reduction=reduction
                )

                # 处理输出格式
                embed = hidden_states[0]  # 单个样本的嵌入
                if self.use_half_precision:
                    embed = embed.float()  # 转换为float32便于后续处理
                if return_numpy:
                    embed = embed.cpu().numpy()

                embeddings.append(embed)

        return embeddings if len(embeddings) > 1 else embeddings[0]

    def batch_extract_embedding(
            self,
            sequences: List[str],
            batch_size: int = 8,
            reduction: Optional[str] = "mean",
            return_numpy: bool = True
    ) -> Union[List[torch.Tensor], List[np.ndarray]]:
        """
        批量提取嵌入特征（适用于大量序列）

        Args:
            sequences: 蛋白质序列列表
            batch_size: 批处理大小
            reduction: 嵌入聚合方式
            return_numpy: 是否转换为numpy数组

        Returns:
            嵌入特征列表
        """
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_embeds = self.extract_embedding(
                batch,
                reduction=reduction,
                return_numpy=return_numpy
            )
            all_embeddings.extend(batch_embeds)
        return all_embeddings


# 使用示例
if __name__ == "__main__":
    # 初始化提取器（替换为实际模型路径）
    extractor = SaProtEmbeddingExtractor(
        model_path="/path/to/SaProt_650M_AF2",
        device="cuda",
        use_half_precision=True
    )

    # 示例序列（含结构标记#）
    sample_sequence = "M#EvVpQpL#VyQdYaKv"

    # 提取单个序列嵌入
    single_embedding = extractor.extract_embedding(
        sequence=sample_sequence,
        reduction="mean"
    )
    print(f"单个序列嵌入形状: {single_embedding.shape}")

    # 批量提取嵌入
    batch_sequences = [sample_sequence, "A#BcDeFgHiJkLmNoP"]
    batch_embeddings = extractor.batch_extract_embedding(
        sequences=batch_sequences,
        batch_size=2
    )
    print(f"批量嵌入数量: {len(batch_embeddings)}")
