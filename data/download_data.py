from datasets import load_dataset

def dataset_spliter():
    ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")
    raw_train_ds = ds["train"]   # 训练集（约20k样本）
    unique_targets = list(set(raw_train_ds["Target_ID"]))
    import random
    random.seed(42)
    val_targets = random.sample(unique_targets, k=int(len(unique_targets)*0.1))
    train_ds = raw_train_ds.filter(lambda x: x["Target_ID"] not in val_targets)
    val_ds = raw_train_ds.filter(lambda x: x["Target_ID"] in val_targets)
    print(f"训练集样本数：{len(train_ds)}")  # 约22.5k
    print(f"验证集样本数：{len(val_ds)}")    # 约2.5k
    return train_ds, val_ds