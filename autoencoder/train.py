import yaml
import pytorch_lightning as pl
from pyL_modules import PyLDataModule, PyLModel
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary


# Global variables
with open("./configs/config_utils.yaml", "r") as file:
    config_utils = yaml.safe_load(file)
    config_utils = {k: v["value"] for k, v in config_utils.items()}


def train():
    model_dir = config_utils["training_config"]["model_dir"]
    exp_name = config_utils["training_config"]["experiment_name"]

    print(f"Model Directory: {model_dir}")
    print(f"Experiment Name: {exp_name}")

    # pl.seed_everything(42, workers=True)

    print("Initializing Data Module...")
    dataset = PyLDataModule()

    print("Initializing Model...")
    model = PyLModel()
    model_summary = ModelSummary(max_depth=2)

    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_dir,
        filename=f"{exp_name}_best_val_loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        max_epochs=100,
        callbacks=[ckpt_callback, model_summary],
        gradient_clip_val=1.0,
        sync_batchnorm=True,
        logger=pl.loggers.TensorBoardLogger(model_dir, name=exp_name),
        enable_progress_bar=True,
    )

    print("Starting Training...")
    trainer.fit(model, dataset)

    print("Training Complete!")


if __name__ == "__main__":
    train()
