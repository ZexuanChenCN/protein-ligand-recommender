import os
import logging
import warnings
import matplotlib.pyplot as plt
import json


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
for logger_name in ["urllib3", "fsspec", "filelock", "transformers", "datasets"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.set_device(0)
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from model.dual_tower import CLIPStyleDualTower
from data.dataloader import get_dataloaders


class MetricsLoggingCallback(Callback):
    def __init__(self, save_dir='./training_logs'):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_margin_score': [],
            'lr': [],
            'temperature': []
        }

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_train_metrics = {
            'epoch': trainer.current_epoch + 1,
            'train_loss': 0.0,
            'train_acc': 0.0,
            'lr': 0.0,
            'temperature': 0.0
        }

    def _get_metric_with_suffix(self, callback_metrics, metric_name):
        if metric_name in callback_metrics:
            val = callback_metrics[metric_name]
            return val.item() if isinstance(val, torch.Tensor) else float(val)
        suffix_name = f"{metric_name}_dataloader_idx_0"
        if suffix_name in callback_metrics:
            val = callback_metrics[suffix_name]
            return val.item() if isinstance(val, torch.Tensor) else float(val)
        return 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1

        # 读取指标
        train_loss = self._get_metric_with_suffix(trainer.callback_metrics, 'train_loss_step')
        train_acc = self._get_metric_with_suffix(trainer.callback_metrics, 'train_acc')
        temperature = self._get_metric_with_suffix(trainer.callback_metrics, 'temperature_step')
        val_loss = self._get_metric_with_suffix(trainer.callback_metrics, 'val_loss')
        val_acc = self._get_metric_with_suffix(trainer.callback_metrics, 'val_acc')
        val_margin = self._get_metric_with_suffix(trainer.callback_metrics, 'val_margin_score')


        optimizer = pl_module.optimizers()
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0


        if current_epoch not in self.metrics['epoch']:
            self.metrics['epoch'].append(current_epoch)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['val_margin_score'].append(val_margin)
            self.metrics['lr'].append(current_lr)
            self.metrics['temperature'].append(temperature)

            print(f"\n[Epoch {current_epoch}] "
                  f"train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}, "
                  f"val_margin={val_margin:.4f}")

    def save_metrics(self, filename='metrics.json'):
        with open(os.path.join(self.save_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"指标已保存至：{os.path.join(self.save_dir, filename)}")

    def plot_metrics(self, save_filename='training_metrics.png'):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        epochs = self.metrics['epoch']
        if not epochs:
            print("无指标数据，跳过绘图")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 损失曲线
        ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, self.metrics['val_loss'], 'b--', label='验证损失')
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练/验证损失变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 准确率曲线
        ax2.plot(epochs, self.metrics['train_acc'], 'r-', label='训练准确率')
        ax2.plot(epochs, self.metrics['val_acc'], 'r--', label='验证准确率')
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('准确率')
        ax2.set_title('训练/验证准确率变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Margin Score
        ax3.plot(epochs, self.metrics['val_margin_score'], 'orange', label='验证Margin Score')
        ax3.set_xlabel('训练轮数')
        ax3.set_ylabel('Margin Score')
        ax3.set_title('验证集Margin Score变化')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Temperature
        ax4.plot(epochs, self.metrics['temperature'], 'purple', label='Temperature')
        ax4.set_xlabel('训练轮数')
        ax4.set_ylabel('Temperature')
        ax4.set_title('Temperature变化趋势')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化图表已保存至：{os.path.join(self.save_dir, save_filename)}")

def main():
    # 核心配置（适配SaProt）
    config = {
        "saprot_config_path": "../models/SaProt_1.3B_AFDB_OMG_NCBI",  # 本地SaProt模型路径（关键！需替换为你的实际路径）
        "chemberta_model_name": "DeepChem/ChemBERTa-77M-MLM",
        "chemberta_hidden_dim": 384,
        "init_temperature": 0.2,
        "hard_neg_weight": 1.1,
        "num_neg_samples": 4,
        "pos_threshold": 5.0,
        "lr": 5e-5,
        "weight_decay": 0.03,
        "batch_size": 16,
        "num_epochs": 20,
        "num_workers": 0,
        "margin": 0.3,
        "dropout": 0.2,
        "proj_dim": 256,
        "filter_min_samples": 5
    }

    # 初始化模型（添加SaProt路径配置）
    print("初始化适配SaProt的双塔模型...")
    model = CLIPStyleDualTower(
        saprot_config_path=config["saprot_config_path"],
        protein_embed_dim=256,
        proj_dim=config["proj_dim"],
        init_temperature=config["init_temperature"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        hard_neg_weight=config["hard_neg_weight"],
        margin=config["margin"],
        dropout=config["dropout"]
    )

    # 获取DataLoader
    train_loader, val_loader = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pos_threshold=config["pos_threshold"],
        margin=config["margin"],
        num_neg_samples=config["num_neg_samples"]
    )

    # 回调函数
    metrics_callback = MetricsLoggingCallback(save_dir='./training_logs')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="./checkpoints",
        filename="saprot_clip_tower_best",
        verbose=True,
        save_weights_only=False,
        every_n_epochs=1
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True
    )


    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        precision="32",
        log_every_n_steps=5,
        enable_progress_bar=True,
        callbacks=[metrics_callback, checkpoint_callback, early_stopping_callback],
        enable_model_summary=True,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        fast_dev_run=False,
        accumulate_grad_batches=2,  #
        enable_checkpointing=True
    )


    print("="*50)
    print(f"开始训练SaProt双塔模型（{config['num_epochs']}个epoch）")
    print("="*50)
    trainer.fit(model, train_loader, val_loader)


    metrics_callback.save_metrics()
    metrics_callback.plot_metrics()

    print("\n训练完成！最佳模型保存至 ./checkpoints/best_model.ckpt")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()