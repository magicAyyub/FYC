import asyncio
import json
import random
from typing import Dict, List, Set
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Rectangle(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float = 0.8


class DetectionResponse(BaseModel):
    rectangles: List[Rectangle]
    timestamp: str
    fps: int
    model: str


app = FastAPI(
    title="Vision AI Backend",
    description="API backend pour la détection de scènes en temps réel",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis le frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage des connexions WebSocket actives
active_connections: Set[WebSocket] = set()

# Modèles disponibles et leurs scènes associées
SCENES_BY_MODEL = {
    "scene-v1": ["Indoor", "Outdoor", "Office", "Street", "Room", "Kitchen"],
    "places365": ["Restaurant", "Bedroom", "Living Room", "Park", "Beach", "Forest"],
    "urban-v2": ["Street", "Building", "Parking", "Sidewalk", "Crosswalk", "Plaza"],
    "nature-v1": ["Forest", "Mountain", "Lake", "Field", "Garden", "Sky"],
}


def generate_dummy_rectangle(
    container_width: float = 640,
    container_height: float = 480,
    model: str = "scene-v1"
) -> Rectangle:
    """Génère un rectangle factice pour simuler la détection."""
    # Dimensions du rectangle
    width = 80 + random.random() * 100
    height = 80 + random.random() * 100
    
    # Position aléatoire dans le conteneur
    x = random.random() * (container_width - width)
    y = random.random() * (container_height - height)
    
    # Label aléatoire selon le modèle
    scenes = SCENES_BY_MODEL.get(model, SCENES_BY_MODEL["scene-v1"])
    label = random.choice(scenes)
    
    # Confiance aléatoire entre 0.6 et 0.95
    confidence = 0.6 + random.random() * 0.35
    
    return Rectangle(
        x=x,
        y=y,
        width=width,
        height=height,
        label=label,
        confidence=confidence
    )


@app.get("/")
async def root():
    """Point d'entrée de l'API."""
    return {
        "message": "Vision AI Backend",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/models")
async def get_available_models():
    """Retourne la liste des modèles disponibles."""
    return {
        "models": list(SCENES_BY_MODEL.keys()),
        "default": "scene-v1"
    }


@app.get("/detect")
async def get_detection(
    width: float = 640,
    height: float = 480,
    model: str = "scene-v1"
):
    """Endpoint REST pour obtenir une détection unique."""
    rectangle = generate_dummy_rectangle(width, height, model)
    
    response = DetectionResponse(
        rectangles=[rectangle],
        timestamp=datetime.now().isoformat(),
        fps=0,  # Pas de FPS pour l'endpoint REST
        model=model
    )
    
    return response


@app.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):
    """WebSocket pour la détection en temps réel."""
    await websocket.accept()
    active_connections.add(websocket)
    
    # Paramètres par défaut
    container_width = 640
    container_height = 480
    model = "scene-v1"
    update_interval = 0.5  # 500ms par défaut
    is_running = False
    
    try:
        # Boucle principale de communication
        while True:
            try:
                # Vérifier s'il y a des messages du client
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                message = json.loads(data)
                
                # Traiter les commandes du client
                if message.get("type") == "start":
                    is_running = True
                    container_width = message.get("width", 640)
                    container_height = message.get("height", 480)
                    model = message.get("model", "scene-v1")
                    update_interval = message.get("interval", 500) / 1000  # Convert ms to seconds
                    
                elif message.get("type") == "stop":
                    is_running = False
                    
                elif message.get("type") == "config":
                    # Mise à jour de la configuration
                    container_width = message.get("width", container_width)
                    container_height = message.get("height", container_height)
                    model = message.get("model", model)
                    update_interval = message.get("interval", update_interval * 1000) / 1000
                    
            except asyncio.TimeoutError:
                # Pas de message reçu, continuer
                pass
            
            # Envoyer une détection si le système est en marche
            if is_running:
                rectangle = generate_dummy_rectangle(container_width, container_height, model)
                
                response = DetectionResponse(
                    rectangles=[rectangle],
                    timestamp=datetime.now().isoformat(),
                    fps=int(1 / update_interval) if update_interval > 0 else 0,
                    model=model
                )
                
                await websocket.send_text(response.model_dump_json())
            
            # Attendre avant la prochaine itération
            await asyncio.sleep(update_interval if is_running else 0.1)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Erreur WebSocket: {e}")
    finally:
        active_connections.discard(websocket)


@app.get("/status")
async def get_status():
    """Retourne le statut du serveur."""
    return {
        "status": "running",
        "active_connections": len(active_connections),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )