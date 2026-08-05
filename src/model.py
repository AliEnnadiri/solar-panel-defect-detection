import torch
import torch.nn as nn
from torchvision import models


class SolarPanelClassifier(nn.Module):
    """
    Modèle de classification d'images pour panneaux solaires basé sur ResNet34 avec Transfer Learning.
    La tête de classification est personnalisée avec BatchNorm, Linear et Dropout.
    """
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super(SolarPanelClassifier, self).__init__()
        
        # 1. Chargement du backbone ResNet34 pré-entraîné
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # 2. Gel sélectif des paramètres du backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Débloquer le dernier bloc résiduel (layer4) pour permettre le fine-tuning
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # 3. Récupération du nombre de features en entrée du dernier layer
        in_features = self.backbone.fc.in_features
        
        # 4. Remplacement de la tête de classification par notre propre Custom MLP
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant (Forward pass).
        Args:
            x (torch.Tensor): Tensor d'images de taille (Batch_Size, 3, H, W)
        Returns:
            torch.Tensor: Logits non-normalisés de taille (Batch_Size, num_classes)
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Test rapide de l'architecture (Vérification des dimensions de sortie)
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch de 2 images de 224x224
    model = SolarPanelClassifier(num_classes=4)
    output = model(dummy_input)
    print(f"Test du Modèle Réussi !")
    print(f"Dimension de l'entrée  : {dummy_input.shape}")
    print(f"Dimension de la sortie : {output.shape} (Attendu: [2, 4])")