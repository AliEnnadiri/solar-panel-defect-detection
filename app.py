import os
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from torchvision import transforms

# Import de notre modèle PyTorch custom
from src.model import SolarPanelClassifier

# ---------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Inspection Intelligente de Panneaux Solaires",
    page_icon="☀️",
    layout="wide"
)

# ---------------------------------------------------------
# Chargement du modèle PyTorch entraîné
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model(checkpoint_path: str, num_classes: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarPanelClassifier(num_classes=num_classes, freeze_backbone=False)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, device, True
    else:
        return model, device, False

CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pth")
model, device, is_loaded = load_trained_model(CHECKPOINT_PATH)

# Transformations pour l'inférence (identiques au jeu de validation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASSES = ["Cellule Intacte (Propre)", "Cellule Défectueuse (Anomalie)"]

# ---------------------------------------------------------
# Barre latérale (Sidebar) - Présentation du Projet
# ---------------------------------------------------------
st.sidebar.title("☀️ Control Center")
st.sidebar.markdown("**Projet Deep Learning PyTorch**")
st.sidebar.markdown("---")
st.sidebar.write("**Domaine :** Maintenance Prédictive PV")
st.sidebar.write("**Dataset :** ELPV Dataset (ZAE Bayern)")
st.sidebar.write("**Backbone :** ResNet34 (Transfer Learning)")

st.sidebar.markdown("---")
st.sidebar.subheader("Statut du Modèle")
if is_loaded:
    st.sidebar.success("✅ Modèle entraîné chargé (`best_model.pth`)")
else:
    st.sidebar.warning("⚠️ Entraînement en cours... (Modèle non prêt)")

# ---------------------------------------------------------
# Zone Principale
# ---------------------------------------------------------
st.title("🔍 Diagnostic & Inspection Automatisée de Panneaux Solaires")
st.markdown("""
Cette application permet d'analyser en temps réel l'état de santé des cellules photovoltaïques 
par Imagerie Électroluminescente (EL) grâce à un Réseau de Neurones Convolutif PyTorch.
""")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Importer une image d'inspection")
    uploaded_file = st.file_uploader(
        "Choisissez une image de cellule (Format PNG, JPG, JPEG)...", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image de la cellule soumise à l'inspection", use_container_width=True)

with col2:
    st.subheader("2. Résultats du Diagnostic Deep Learning")
    
    if uploaded_file is not None:
        if not is_loaded:
            st.error("Le modèle n'est pas encore sauvegardé. Veuillez attendre la fin de l'entraînement de `src/train.py`.")
        else:
            with st.spinner("Analyse par le réseau de neurones..."):
                # Prétraitement de l'image
                img_tensor = val_transform(image).unsqueeze(0).to(device)
                
                # Inférence
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class].item() * 100

                # Affichage du diagnostic principal
                if predicted_class == 0:
                    st.success(f"### Verdict : **{CLASSES[0]}**")
                else:
                    st.error(f"### Verdict : **{CLASSES[1]}**")
                
                st.metric(label="Indice de Confiance de la Prédiction", value=f"{confidence:.2f} %")

                # Histogramme des probabilités Plotly
                fig = go.Figure(go.Bar(
                    x=[probabilities[0].item() * 100, probabilities[1].item() * 100],
                    y=CLASSES,
                    orientation='h',
                    marker_color=['#2ecc71', '#e74c3c']
                ))
                fig.update_layout(
                    title="Distribution des Probabilités",
                    xaxis_title="Confiance (%)",
                    yaxis_title="Classe",
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez charger une image dans le panneau de gauche pour exécuter le diagnostic.")