#!/bin/bash

# Script de démarrage du backend Vision AI

echo "🚀 Démarrage du backend Vision AI..."

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Créer un environnement virtuel s'il n'existe pas
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

# Démarrer le serveur
echo "Démarrage du serveur FastAPI sur http://localhost:8000"
echo "WebSocket disponible sur ws://localhost:8000/ws/detect"
echo "Documentation API: http://localhost:8000/docs"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter le serveur"

python main.py