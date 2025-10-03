# Vision AI - Étape 2 : Backend Connecté

Application de détection de scènes en temps réel avec backend FastAPI et frontend Next.js.

## 🚀 Démarrage rapide

### 1. Backend (Terminal 1)
```bash
cd backend
./start.sh
```

### 2. Frontend (Terminal 2)
```bash
npm run dev
```

### 3. Tester l'application
- Ouvrir http://localhost:3000
- Vérifier que l'indicateur "Backend" est vert (connecté)
- Cliquer sur le bouton play pour démarrer la détection
- Observer les rectangles animés provenant du backend

## 📁 Structure du projet

```
labello/
├── app/                    # Frontend Next.js
│   └── page.tsx           # Interface principale
├── backend/               # Backend FastAPI
│   ├── main.py           # Serveur principal
│   ├── requirements.txt  # Dépendances Python
│   └── start.sh         # Script de démarrage
├── components/           # Composants UI
└── ...
```

## 🔌 Communication Frontend ↔ Backend

### WebSocket (Temps réel)
- **URL**: `ws://localhost:8000/ws/detect`
- **Messages**: JSON pour start/stop/config
- **Réponses**: Rectangles avec coordonnées et labels

### REST API (Informations)
- **Documentation**: http://localhost:8000/docs
- **Modèles**: http://localhost:8000/models
- **Statut**: http://localhost:8000/status

## 🎯 Fonctionnalités actuelles

✅ **Étape 1** - Prototype frontend
- Vidéo temps réel
- Interface utilisateur complète
- Paramètres avancés

✅ **Étape 2** - Backend connecté
- Serveur FastAPI avec WebSocket
- Communication bidirectionnelle
- Rectangles générés côté serveur
- Gestion de la reconnexion automatique

🔄 **Prochaine étape** - IA basique
- Intégration modèle PyTorch
- Vraie détection d'objets/scènes

## 🛠️ Paramètres configurables

Via l'interface utilisateur :
- **Modèle** : Type de scènes à détecter
- **Intervalle** : Fréquence de mise à jour (100-2000ms)
- **Seuil de confiance** : Filtre des détections
- **Affichage des labels** : Voir/masquer les textes

## 🐛 Dépannage

### Backend déconnecté
1. Vérifier que le backend tourne sur le port 8000
2. Vérifier les logs dans le terminal backend
3. L'app tente une reconnexion automatique

### Erreurs de caméra
- Utiliser HTTPS pour mobile : https://localhost:3000
- Autoriser l'accès caméra dans le navigateur

### Performance
- Ajuster l'intervalle de mise à jour
- Choisir le modèle adapté à l'environnement

## 📊 Métriques

L'interface affiche :
- **FPS** : Fréquence de détection effective
- **Statut backend** : Connexion WebSocket
- **Statut caméra** : Avant/Arrière

## 🔄 Évolution

Cette étape 2 prépare l'intégration d'un vrai modèle d'IA :
- Architecture modulaire backend/frontend
- Communication temps réel optimisée
- Interface utilisateur prête pour paramètres avancés