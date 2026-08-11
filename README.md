# ☀️ Inspection Automatisée de Panneaux Solaires par Deep Learning

Projet de fin de module en **Vision par Ordinateur** & **Deep Learning** axé sur la maintenance prédictive des installations photovoltaïques. Cette application permet d'analyser des images d'Électroluminescence (EL) pour détecter les micro-fissures et défauts internes sur des cellules solaires.

---

## 📌 Fonctionnalités Clés

* **Modèle de Classification PyTorch :** Transfer Learning basé sur l'architecture **ResNet34**.
* **Prétraitement & Augmentation :** Redimensionnement (224x224), normalisation ImageNet, et flips aléatoires.
* **Interface Web Interactive :** Application développée avec **Streamlit** pour un diagnostic en temps réel.
* **Visualisation Dynamique :** Graphiques de distribution des probabilités générés par **Plotly**.

---

## 📊 Performances du Modèle

Le modèle a été entraîné sur le jeu de données académique **ELPV Dataset** (ZAE Bayern) contenant 2 624 images.

| Époque | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Statut |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1/5 | 0.6320 | 66.79% | 0.5118 | 73.90% | Sauvegardé |
| 2/5 | 0.5594 | 71.84% | 0.5102 | 75.43% | Sauvegardé |
| 3/5 | 0.5595 | 71.61% | 0.4783 | 77.52% | Sauvegardé |
| 4/5 | 0.5337 | 72.84% | 0.4622 | 76.38% | Non retenu |
| **5/5** | **0.5349** | **72.08%** | **0.4542** | **78.10%** | **Meilleur Modèle (`best_model.pth`)** |

* **Précision maximale atteinte :** 78.10% en validation.

---

## 🛠️ Structure du Projet

```text
solar_panel_inspection/
├── checkpoints/
│   └── best_model.pth             # Poids du meilleur modèle PyTorch
├── data/
│   └── raw/                       # Images du dataset ELPV
├── src/
│   ├── dataset.py                 # Class PyTorch Dataset & Data Augmentation
│   ├── model.py                   # Custom ResNet34 Architecture
│   └── train.py                   # Script d'entraînement et validation
├── app.py                         # Application Web Streamlit
├── .gitignore                     # Exclusion des fichiers volumineux
└── README.md                      # Documentation du projet


🚀 Installation & Exécution Locale
1. Cloner le dépôt

git clone [https://github.com/AliEnnadiri/solar-panel-defect-detection.git](https://github.com/AliEnnadiri/solar-panel-defect-detection.git)
cd solar-panel-defect-detection

2. Créer et activer l'environnement virtuel

# Sur Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
3. Installer les dépendances

pip install torch torchvision streamlit plotly pillow pandas
4. Lancer l'application Web Streamlit

streamlit run app.py


👥 Auteurs & Encadrement

Étudiants : Ali ENNADIRI et Khalid Amouzg

Encadrant : Professeur HANOUNE

Année Académique : 2025 - 2026