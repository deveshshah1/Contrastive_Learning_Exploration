import torch
import os
import yaml
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pyL_modules import PyLModel
from torch_dataset import CustomDataset

with open("./configs/config_utils.yaml", "r") as file:
    config_utils = yaml.safe_load(file)
    config_utils = {k: v["value"] for k, v in config_utils.items()}


def predict():
    # Load the dataset 
    print("Loading dataset...")
    df = pd.read_csv(config_utils["dataset_labels_path"])
    dataset = CustomDataset(stage="ALL", augment=False, size=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    assert len(df) == len(dataset), "Dataset length mismatch with labels CSV."

    model_path = os.path.join(config_utils["model_dir"], f"{config_utils["exp_name"]}_best_val_loss.ckpt")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")

    model = PyLModel.load_from_checkpoint(
        model_path,
        map_location=device,
    ).to(device)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        enable_progress_bar=True,
    )

    print("Starting prediction...")
    out_batches = trainer.predict(model, dataloaders=dataloader)

    sample_id, emb = [], []
    for batch in out_batches:
        sample_id.append(batch["sample_id"].cpu().numpy().astype(np.int32))
        emb.extend(batch["embeddings"].cpu().numpy().astype(np.float32))
    
    sample_id = np.concatenate(sample_id)
    emb = np.concatenate(emb).tolist()

    df_tomerge = pd.DataFrame({
        "sample_id": sample_id,
        "embeddings": emb
    })
    assert df_tomerge["sample_id"].is_unique, "Duplicate sample_ids in prediction output."

    df = pd.merge(df, df_tomerge, on="sample_id", how="left")
    df.to_parquet("results.parquet", index=False)
    return


if __name__ == "__main__":
    predict()
