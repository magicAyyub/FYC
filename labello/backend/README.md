# Vision AI Backend

Backend FastAPI pour l'application Vision AI - Étape 2 de la roadmap.

## 🚀 Démarrage rapide

```bash
cd backend
./start.sh
```

Le serveur sera disponible sur :
- **API REST** : http://localhost:8000
- **WebSocket** : ws://localhost:8000/ws/detect
- **Documentation** : http://localhost:8000/docs

## 📡 Endpoints

### REST API

- `GET /` - Informations sur l'API
- `GET /models` - Liste des modèles disponibles
- `GET /detect` - Détection unique
  - `width` (float) : Largeur du conteneur
  - `height` (float) : Hauteur du conteneur  
  - `model` (string) : Modèle à utiliser
- `GET /status` - Statut du serveur

### WebSocket

- `ws://localhost:8000/ws/detect` - Détection en temps réel

#### Messages WebSocket

**Démarrer la détection :**
```json
{
  "type": "start",
  "width": 640,
  "height": 480,
  "model": "scene-v1",
  "interval": 500
}
```

**Arrêter la détection :**
```json
{
  "type": "stop"
}
```

**Mettre à jour la configuration :**
```json
{
  "type": "config",
  "width": 800,
  "height": 600,
  "model": "places365",
  "interval": 300
}
```

**Réponse du serveur :**
```json
{
  "rectangles": [
    {
      "x": 123.45,
      "y": 67.89,
      "width": 120.0,
      "height": 90.0,
      "label": "Indoor",
      "confidence": 0.85
    }
  ],
  "timestamp": "2025-10-03T10:30:00.123456",
  "fps": 2,
  "model": "scene-v1"
}
```

## 🎯 Modèles disponibles

- `scene-v1` : Scènes génériques (Indoor, Outdoor, Office, etc.)
- `places365` : Lieux spécifiques (Restaurant, Bedroom, Park, etc.)
- `urban-v2` : Environnements urbains (Street, Building, Parking, etc.)
- `nature-v1` : Environnements naturels (Forest, Mountain, Lake, etc.)

## 🔧 Configuration

Les paramètres peuvent être ajustés via WebSocket :
- **Intervalle** : Délai entre chaque détection (100-2000ms)
- **Dimensions** : Taille du conteneur vidéo
- **Modèle** : Type de scènes à détecter

## 📦 Installation manuelle

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Démarrer le serveur
python main.py
```