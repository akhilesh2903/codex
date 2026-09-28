import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np

from src.data.dataset import DRDataset
from src.preprocessing.transforms import get_transforms
from src.classification.model import get_model

def load_config(config_path):
    with open(config_path, "r") as f:
         return yaml.safe_load(f)

def train(config_path):
    config = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    
    # Load splits
    split_dir = "data/splits"
    if not os.path.exists(os.path.join(split_dir, "train.csv")):
        print("Splits not found. Please run prepare_dataset.py first. (Demo mode bypass)")
        return
        
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    
    dataset_root = config["dataset"]["root"]
    image_dir = os.path.join(dataset_root, "train_images")
    
    train_transform = get_transforms(image_size=config['image']['size'], mode="train")
    val_transform = get_transforms(image_size=config['image']['size'], mode="val")
    
    train_dataset = DRDataset(train_df, image_dir, transform=train_transform)
    val_dataset = DRDataset(val_df, image_dir, transform=val_transform)
    
    # Class imbalance handling (weighted sampler)
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_df['diagnosis']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = get_model(num_classes=config['classification']['num_classes'], model_name=config['classification']['model'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    os.makedirs("models/classification", exist_ok=True)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/classification/best_model.pth")
            print("Model saved!")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DR Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    train(args.config)
