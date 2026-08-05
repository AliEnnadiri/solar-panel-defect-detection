import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ELPVSolarDataset, get_data_transforms
from model import SolarPanelClassifier


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    return epoch_loss, epoch_acc.item()


def run_training(
    data_dir: str = os.path.join("data", "raw", "elpv-dataset-master"),
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
    checkpoint_path: str = os.path.join("checkpoints", "best_model.pth")
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Matériel utilisé : {device}")

    train_tf, val_tf = get_data_transforms()
    
    print(f"--> Chargement du dataset depuis : {data_dir}")
    full_dataset = ELPVSolarDataset(base_dir=data_dir, transform=train_tf)
    num_classes = len(full_dataset.classes)
    
    print(f"--> Classes détectées ({num_classes}) : {full_dataset.classes}")
    print(f"--> Nombre total d'images chargées : {len(full_dataset)}")

    # Split Train / Validation (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SolarPanelClassifier(num_classes=num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print("\n--- Début de l'Entraînement ---")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Époque [{epoch+1}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% "
              f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    --> Modèle sauvegardé dans {checkpoint_path} ({val_acc*100:.2f}%)")

    print(f"\n--- Entraînement terminé. Meilleure Précision : {best_val_acc*100:.2f}% ---")


if __name__ == "__main__":
    print("=== Démarrage du script d'entraînement ===")
    run_training(epochs=5, batch_size=16)