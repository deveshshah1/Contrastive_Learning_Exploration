import yaml
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
)

with open("./configs/config_utils.yaml", "r") as file:
    config_utils = yaml.safe_load(file)
    config_utils = {k: v["value"] for k, v in config_utils.items()}


class CustomDataset(Dataset):
    def __init__(self, stage, augment=False, size=128):
        super(CustomDataset, self).__init__()
        self.stage = stage
        self.augment = augment
        self.images_base_path = config_utils["dataset_images_base_path"]

        self.data = pd.read_csv(config_utils["dataset_labels_path"])
        if stage != "ALL":
            self.data = self.data[self.data["stage"] == stage]

        self.default_transforms = Compose(
            [Resize((size, size)), ToTensor(), Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )

        self.augment_transforms = Compose(
            [
                Resize((size, size)),
                RandomHorizontalFlip(),
                RandomRotation(degrees=15),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ToTensor(),
                Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

        print("Loaded dataset for stage:", self.stage)
        print("Number of samples:", len(self.data))
        print("Number of unique labels: ", self.data["label"].nunique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.images_base_path}/{row['image_path']}"
        label = row["label"]

        image = Image.open(image_path).convert("RGB")

        if self.augment:
            image = self.augment_transforms(image)
        else:
            image = self.default_transforms(image)

        return {"image": image, "label": label}


if __name__ == "__main__":
    stage = "train"
    idx = 0

    dataset = CustomDataset(stage=stage)

    item = dataset[idx]
    print(f"Item at index {idx}:")
    print("Image shape:", item["image"].shape)
    print("Label:", item["label"])
