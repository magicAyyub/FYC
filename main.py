import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm # Importation de tqdm pour la barre de progression
from helper import get_device

# --- Configuration et Constantes ---
# Utilisez le dictionnaire ID_TO_TRAINID pour mapper les IDs Cityscapes non séquentiels
# aux IDs séquentiels pour l'entraînement (0-18) et marquer 255 comme 'ignorer'.
ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 
    9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 
    17: 5, 18: 6, 19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 
    26: 14, 27: 15, 28: 16, 29: 255, 30: 255, 31: 17, 32: 18, 33: 255
}

# 19 classes d'évaluation + 1 (l'ID 255 à ignorer)
NUM_CLASSES = 19
# Hyperparamètres de Formation
IMAGE_SIZE = (512, 1024)
BATCH_SIZE = 4
EPOCHS = 10 
LEARNING_RATE = 0.0001
DEVICE = get_device()


# --- Custom Dataset Class ---

class CityscapesDataset(Dataset):
    """
    Classe Dataset pour charger les images (leftImg8bit) et les masques (gtFine_labelIds)
    et appliquer la conversion ID_TO_TRAINID.
    """
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.id_to_trainid = ID_TO_TRAINID
        
        self.img_paths = self._get_paths(os.path.join(root_dir, 'leftImg8bit', split))
        self.mask_paths = [
            p.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            for p in self.img_paths
        ]
        
        # Valider que tous les fichiers de masques existent
        valid_pairs = []
        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Mask not found for {img_path}")
                print(f"Expected: {mask_path}")
        
        self.img_paths = [p[0] for p in valid_pairs]
        self.mask_paths = [p[1] for p in valid_pairs]

    def _get_paths(self, split_dir):
        """Recherche récursivement tous les chemins d'image dans le répertoire."""
        # Filtrer uniquement les fichiers _leftImg8bit.png
        all_paths = glob.glob(os.path.join(split_dir, '*', '*_leftImg8bit.png'))
        return all_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 2. Load Mask (Ground Truth)
        mask_path = self.mask_paths[idx]
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        
        # 3. Convert Cityscapes IDs to sequential Train IDs (0-18 or 255)
        target = mask.copy()
        for city_id, train_id in self.id_to_trainid.items():
            target[mask == city_id] = train_id
        
        target = Image.fromarray(target) 

        # 4. Apply Transforms (image and mask)
        if self.transforms:
            # Pour l'image : Resize, ToTensor, Normalize
            image = self.transforms(image)
            # Pour le masque : Resize (pour correspondre à la taille de l'image), ToTensor (non normalisant)
            target = transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)(target)
            # Convertir en tensor, puis multiplier par 255 pour récupérer les IDs entiers (0-18, 255)
            target = transforms.ToTensor()(target).squeeze() * 255 
        
        # Le masque doit être de type LongTensor pour CrossEntropyLoss
        target = target.long()
        
        return image, target


# --- Transforms et Configuration du Modèle ---

def get_transforms(image_size=(512, 1024)):
    """Définit les transformations de normalisation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),         
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]   
        )
    ])

def setup_model():
    """Charge le modèle FPN/MobileNetV2 et configure les composants d'entraînement."""
    print(f"Utilisation du périphérique: {DEVICE}")

    # Utilisation de FPN/MobileNetV2, une architecture rapide et performante, pour respecter
    # l'exigence de vitesse du projet (alternative à BiSeNetV2 dans SMP).
    model = smp.FPN( 
        encoder_name="mobilenet_v2",  
        encoder_weights="imagenet",   
        in_channels=3,
        classes=NUM_CLASSES,          
        activation=None,              
    )
    model.to(DEVICE)

    # Fonction de perte: ignore_index=255 pour ignorer les pixels non étiquetés
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Métrique: IoU (Intersection sur Union), standard pour la segmentation
    # Utilisation de torchmetrics pour le calcul du mIoU
    from torchmetrics.classification import MulticlassJaccardIndex
    metrics = {
        'iou': MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255, average='macro')
    }

    return model, loss_fn, optimizer, metrics

# --- Fonctions d'Entraînement et de Validation ---

def train_epoch(model, dataloader, loss_fn, optimizer, metrics):
    """Boucle d'entraînement pour une seule époque."""
    model.train()
    total_loss = 0
    metric_values = {k: 0.0 for k in metrics.keys()}

    for images, targets in tqdm(dataloader, desc="Training"):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calcul des métriques
        preds = torch.argmax(outputs, dim=1)
        for k, metric in metrics.items():
             # Mettre à jour l'état de la métrique pour le lot
             metric.to(DEVICE)(preds, targets)
             
    avg_loss = total_loss / len(dataloader)
    
    # Calculer la valeur finale de la métrique pour l'époque
    final_metrics = {k: metric.compute().item() for k, metric in metrics.items()}
    
    # Réinitialiser l'état des métriques pour la prochaine époque
    for metric in metrics.values():
        metric.reset()
        
    return avg_loss, final_metrics

def validate_epoch(model, dataloader, loss_fn, metrics):
    """Boucle de validation."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            for k, metric in metrics.items():
                metric.to(DEVICE)(preds, targets)

    avg_loss = total_loss / len(dataloader)
    
    # Calculer la valeur finale de la métrique pour l'époque
    final_metrics = {k: metric.compute().item() for k, metric in metrics.items()}
    
    # Réinitialiser l'état des métriques
    for metric in metrics.values():
        metric.reset()
        
    return avg_loss, final_metrics


def main():
    """Fonction principale pour initialiser les données et commencer l'entraînement."""
    # Création du dossier pour sauvegarder le modèle
    os.makedirs('checkpoints', exist_ok=True)
    
    # Setup data loaders
    data_dir = 'data/cityscapes'
    train_transforms = get_transforms(IMAGE_SIZE)
    
    # On utilise un dataset réduit pour les tests initiaux si le Cityscapes complet est trop lourd
    train_dataset = CityscapesDataset(data_dir, split='train', transforms=train_transforms)
    val_dataset = CityscapesDataset(data_dir, split='val', transforms=train_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Échantillons d'entraînement: {len(train_dataset)}, Échantillons de validation: {len(val_dataset)}")

    # Setup model, loss, optimizer, and metrics
    model, loss_fn, optimizer, metrics = setup_model()
    best_iou = 0.0

    # Start training loop
    print("Démarrage de l'entraînement...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Époque {epoch}/{EPOCHS} ---")
        
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, metrics)
        print(f"Résultats Entraînement: Loss={train_loss:.4f}, mIoU={train_metrics['iou']:.4f}")

        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, metrics)
        val_iou = val_metrics['iou']
        print(f"Résultats Validation: Loss={val_loss:.4f}, mIoU={val_iou:.4f}")

        # Sauvegarde du meilleur modèle basé sur le mIoU de validation
        if val_iou > best_iou:
            best_iou = val_iou
            model_path = os.path.join('checkpoints', 'best_fpn_mobilenetv2_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Meilleur modèle sauvegardé: {model_path} avec mIoU={best_iou:.4f}")

    print("\nEntraînement terminé.")
    
if __name__ == '__main__':
    main()