import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_dataset import CustomDataset
from model import Autoencoder


class PyLDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(stage="train", augment=True)
        self.val_dataset = CustomDataset(stage="val", augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )


class PyLModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Autoencoder(latent_dim=128)
        self.loss_fxn = torch.nn.MSELoss()
        self.lr = 1e-3
        self.weight_decay = 1e-5

    def on_fit_start(self):
        print(f"[Lightning] Training on device: {self.device}")

    def forward(self, x):
        x_recon, emb = self.model(x)
        return x_recon, emb

    def psnr(recon, target, max_val=1.0):
        mse = torch.mean((recon - target) ** 2)
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        x_recon, emb = self.forward(x)
        loss = self.loss_fxn(x_recon, x)
        self.log("train_loss", loss)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        x_recon, emb = self.forward(x)
        loss = self.loss_fxn(x_recon, x)
        self.log("val_loss", loss)
        psnr_value = self.psnr(x_recon, x)
        self.log("val_psnr", psnr_value, prog_bar=True)
        return loss

    def predict_step(self, batch):
        x, y = batch["image"], batch["label"]
        self.model.eval()
        x_recon, emb = self.forward(x)
        return {"img": x, "img_reconstructed": x_recon, "embedding": emb, "label": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


if __name__ == "__main__":
    # Example usage Dataset
    dataset = PyLDataModule()
    dataset.setup()
    print("Train dataset size:", len(dataset.train_dataset))
    print("Validation dataset size:", len(dataset.val_dataset))

    # Example usage Model
    model = PyLModel(stage="train")
    print(model)
    x = torch.randn(1, 3, 128, 128)  # Example input
    reconstructed, embedding = model(x)
    print(reconstructed.shape, embedding.shape)
