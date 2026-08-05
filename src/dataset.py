import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ELPVSolarDataset(Dataset):
    """
    Dataset PyTorch sur-mesure pour ELPV.
    Lit labels.csv et charge les images associées.
    - 0 : Intacte (probabilité de défaut <= 0.33)
    - 1 : Défectueuse (probabilité de défaut > 0.33)
    """
    def __init__(self, base_dir: str, transform=None):
        self.transform = transform
        
        # Le dossier contenant le module du dataset dans le ZIP GitHub
        self.elpv_data_dir = os.path.join(
            base_dir, "src", "elpv_dataset", "data"
        )
        
        labels_path = os.path.join(self.elpv_data_dir, "labels.csv")
        images_dir = os.path.join(self.elpv_data_dir, "images")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Fichier introuvable : {labels_path}")
        
        # Lecture du fichier CSV d'ELPV (séparateur par espace/tabulation)
        df = pd.read_csv(labels_path, sep=r'\s+', header=None, names=["path", "prob", "type"])
        
        # Binary Classification
        df["label"] = (df["prob"] > 0.33).astype(int)
        
        self.image_paths = [os.path.join(images_dir, os.path.basename(p)) for p in df["path"]]
        self.labels = df["label"].tolist()
        
        self.classes = ["Intact_Cell", "Defective_Cell"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


def get_data_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform