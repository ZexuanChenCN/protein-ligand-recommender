from datasets import load_dataset

def dataset_spliter():
    ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")
    raw_train_ds = ds["train"]   # 训练集（约20k样本）
    # 1. 先获取所有唯一的蛋白ID
    unique_targets = list(set(raw_train_ds["Target_ID"]))

    # 2. 随机选10%的蛋白ID作为验证集的蛋白（冷目标）
    import random
    random.seed(42)  # 固定随机种子，保证结果可复现
    val_targets = random.sample(unique_targets, k=int(len(unique_targets)*0.1))

    # 3. 拆分训练集/验证集：
    # 训练集：蛋白ID不在val_targets中的样本
    train_ds = raw_train_ds.filter(lambda x: x["Target_ID"] not in val_targets)
    # 验证集：蛋白ID在val_targets中的样本
    val_ds = raw_train_ds.filter(lambda x: x["Target_ID"] in val_targets)

    print(f"训练集样本数：{len(train_ds)}")  # 约22.5k
    print(f"验证集样本数：{len(val_ds)}")    # 约2.5k
    return train_ds, val_ds