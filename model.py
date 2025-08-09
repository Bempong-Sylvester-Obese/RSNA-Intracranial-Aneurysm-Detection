# =============================================================================
# CELL 1: IMPORTS AND SETUP
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Machine Learning Libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

# Image Processing
import cv2
from PIL import Image

# Progress bars
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# CELL 2: DATA CONFIGURATION AND PATHS
# =============================================================================

# Configuration
class Config:
    # Data paths
    DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/Data"
    TRAIN_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/Data/train.csv"
    TEST_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/Data/test.csv"
    
    # Model parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Training parameters
    NUM_FOLDS = 5
    NUM_CLASSES = 13
    EARLY_STOPPING_PATIENCE = 10
    
    # Artery labels mapping
    ARTERY_LABELS = {
        'ACA': 'Anterior Communicating Artery',
        'BT': 'Basilar Tip',
        'LACA': 'Left Anterior Cerebral Artery',
        'LIICA': 'Left Infraclinoid Internal Carotid Artery',
        'LMCA': 'Left Middle Cerebral Artery',
        'LPCA': 'Left Posterior Communicating Artery',
        'LSICA': 'Left Supraclinoid Internal Carotid Artery',
        'OPC': 'Other Posterior Circulation',
        'RACA': 'Right Anterior Cerebral Artery',
        'RIICA': 'Right Infraclinoid Internal Carotid Artery',
        'RMCA': 'Right Middle Cerebral Artery',
        'RPCA': 'Right Posterior Communicating Artery',
        'RSICA': 'Right Supraclinoid Internal Carotid Artery'
    }
    
    # Class weights for imbalanced data
    CLASS_WEIGHTS = [1.0] * 13  # Can be adjusted based on class distribution

config = Config()

# =============================================================================
# CELL 3: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """Load and preprocess the training data"""
    print("Loading training data...")
    
    # Load CSV files
    train_df = pd.read_csv(config.TRAIN_CSV)
    print(f"Training data shape: {train_df.shape}")
    
    # Display basic statistics
    print("\nDataset Overview:")
    print(f"Total samples: {len(train_df)}")
    print(f"Patients with aneurysms: {train_df['Aneurysm Present'].sum()}")
    print(f"Aneurysm rate: {train_df['Aneurysm Present'].mean():.3f}")
    
    # Check modality distribution
    print(f"\nModality distribution:")
    print(train_df['Modality'].value_counts())
    
    # Check age and sex distribution
    print(f"\nAge statistics:")
    print(train_df['PatientAge'].describe())
    print(f"\nSex distribution:")
    print(train_df['PatientSex'].value_counts())
    
    return train_df

def get_artery_columns():
    """Get the column names for artery labels"""
    artery_cols = [
        'Left Infraclinoid Internal Carotid Artery',
        'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery',
        'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery',
        'Right Middle Cerebral Artery',
        'Anterior Communicating Artery',
        'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery',
        'Basilar Tip',
        'Other Posterior Circulation'
    ]
    return artery_cols

def analyze_class_distribution(df):
    """Analyze the distribution of aneurysm locations"""
    artery_cols = get_artery_columns()
    
    print("\nAneurysm Location Distribution:")
    for i, col in enumerate(artery_cols):
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"{col}: {count} ({percentage:.2f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(15, 6))
    counts = [df[col].sum() for col in artery_cols]
    plt.bar(range(len(artery_cols)), counts)
    plt.title('Aneurysm Location Distribution')
    plt.xlabel('Artery Location')
    plt.ylabel('Count')
    plt.xticks(range(len(artery_cols)), [col.split()[-1] for col in artery_cols], rotation=45)
    plt.tight_layout()
    plt.show()

# Load data
train_df = load_data()
artery_cols = get_artery_columns()
analyze_class_distribution(train_df)

# =============================================================================
# CELL 4: DATASET CLASS AND DATA LOADERS
# =============================================================================

class BrainAneurysmDataset(Dataset):
    """Custom dataset for brain aneurysm detection"""
    
    def __init__(self, df, data_dir, transform=None, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.artery_cols = get_artery_columns()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        
        # Load image stack
        image_stack = self.load_image_stack(series_id)
        
        # Get labels
        if self.is_train:
            labels = row[self.artery_cols].values.astype(np.float32)
        else:
            labels = np.zeros(len(self.artery_cols), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_stack)
            image_stack = augmented['image']
        
        return {
            'image': image_stack,
            'labels': labels,
            'series_id': series_id,
            'age': row['PatientAge'],
            'sex': row['PatientSex'],
            'modality': row['Modality']
        }
    
    def load_image_stack(self, series_id):
        """Load all images for a given series ID"""
        image_paths = []
        
        # Search for images in all artery directories
        for artery_dir in os.listdir(self.data_dir):
            artery_path = os.path.join(self.data_dir, artery_dir)
            if os.path.isdir(artery_path):
                series_path = os.path.join(artery_path, series_id)
                if os.path.exists(series_path):
                    # Get all PNG files in the series directory
                    png_files = [f for f in os.listdir(series_path) if f.endswith('.png')]
                    png_files.sort()  # Sort to maintain order
                    for png_file in png_files:
                        image_paths.append(os.path.join(series_path, png_file))
                    break  # Found the series, no need to search other directories
        
        if not image_paths:
            # If no images found, create a dummy image
            print(f"Warning: No images found for series {series_id}")
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        
        # Load and stack images
        images = []
        for img_path in image_paths[:50]:  # Limit to first 50 images to avoid memory issues
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                    images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not images:
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        
        # Create a representative image (mean of all images in stack)
        image_stack = np.mean(images, axis=0).astype(np.uint8)
        return image_stack

def get_transforms():
    """Get data augmentation transforms"""
    
    # Training transforms
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Validation transforms
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

# =============================================================================
# CELL 5: MODEL ARCHITECTURE
# =============================================================================

class BrainAneurysmModel(nn.Module):
    """Deep learning model for brain aneurysm detection"""
    
    def __init__(self, num_classes=13, pretrained=True):
        super(BrainAneurysmModel, self).__init__()
        
        # Use EfficientNet as backbone
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Modify the classifier for multi-label classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Add custom head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for the classifier"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =============================================================================
# CELL 6: TRAINING FUNCTIONS
# =============================================================================

class Trainer:
    """Training class for brain aneurysm detection"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_score = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate AUC for each class
        auc_scores = []
        for i in range(all_targets.shape[1]):
            try:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        
        avg_auc = np.mean(auc_scores)
        
        return total_loss / len(self.train_loader), avg_auc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate AUC for each class
        auc_scores = []
        for i in range(all_targets.shape[1]):
            try:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        
        avg_auc = np.mean(auc_scores)
        
        return total_loss / len(self.val_loader), avg_auc
    
    def train(self, num_epochs):
        """Train the model"""
        train_losses = []
        val_losses = []
        train_aucs = []
        val_aucs = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_auc = self.train_epoch()
            
            # Validate
            val_loss, val_auc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_auc > self.best_score:
                self.best_score = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved! AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        return train_losses, val_losses, train_aucs, val_aucs

# =============================================================================
# CELL 7: CROSS-VALIDATION TRAINING
# =============================================================================

def train_with_cross_validation():
    """Train model using cross-validation"""
    
    # Load data
    train_df = load_data()
    artery_cols = get_artery_columns()
    
    # Create target for stratification (presence of any aneurysm)
    train_df['any_aneurysm'] = train_df[artery_cols].sum(axis=1) > 0
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['any_aneurysm'])):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}/{config.NUM_FOLDS}")
        print(f"{'='*50}")
        
        # Split data
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train samples: {len(train_fold)}")
        print(f"Val samples: {len(val_fold)}")
        
        # Create datasets
        train_transform, val_transform = get_transforms()
        
        train_dataset = BrainAneurysmDataset(
            train_fold, config.DATA_DIR, transform=train_transform, is_train=True
        )
        val_dataset = BrainAneurysmDataset(
            val_fold, config.DATA_DIR, transform=val_transform, is_train=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        # Initialize model
        model = BrainAneurysmModel(num_classes=config.NUM_CLASSES).to(device)
        
        # Loss function and optimizer
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Train
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
        train_losses, val_losses, train_aucs, val_aucs = trainer.train(config.NUM_EPOCHS)
        
        # Store best score
        fold_scores.append(trainer.best_score)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Fold {fold+1} - Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_aucs, label='Train AUC')
        plt.plot(val_aucs, label='Val AUC')
        plt.title(f'Fold {fold+1} - AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Print cross-validation results
    print(f"\n{'='*50}")
    print("Cross-Validation Results")
    print(f"{'='*50}")
    print(f"Fold scores: {fold_scores}")
    print(f"Mean CV AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return fold_scores

# =============================================================================
# CELL 8: HYPERPARAMETER OPTIMIZATION
# =============================================================================

def objective(trial):
    """Objective function for Optuna hyperparameter optimization"""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    image_size = trial.suggest_categorical('image_size', [224, 256, 384])
    
    # Update config
    config.LEARNING_RATE = lr
    config.BATCH_SIZE = batch_size
    config.IMAGE_SIZE = image_size
    
    # Load data
    train_df = load_data()
    train_df['any_aneurysm'] = train_df[get_artery_columns()].sum(axis=1) > 0
    
    # Use a smaller subset for hyperparameter optimization
    train_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)
    
    # Split data
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in skf.split(train_df, train_df['any_aneurysm']):
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_transform, val_transform = get_transforms()
        
        train_dataset = BrainAneurysmDataset(
            train_fold, config.DATA_DIR, transform=train_transform, is_train=True
        )
        val_dataset = BrainAneurysmDataset(
            val_fold, config.DATA_DIR, transform=val_transform, is_train=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        # Initialize model
        model = BrainAneurysmModel(num_classes=config.NUM_CLASSES).to(device)
        
        # Loss function and optimizer
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
        
        # Train for fewer epochs during optimization
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
        _, _, _, val_aucs = trainer.train(10)  # Train for 10 epochs only
        
        fold_scores.append(max(val_aucs))
    
    return np.mean(fold_scores)

def optimize_hyperparameters():
    """Run hyperparameter optimization"""
    print("Starting hyperparameter optimization...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    
    return study

# =============================================================================
# CELL 9: INFERENCE AND PREDICTION
# =============================================================================

class Predictor:
    """Class for making predictions on new data"""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = BrainAneurysmModel(num_classes=config.NUM_CLASSES).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Get transforms
        _, self.transform = get_transforms()
    
    def predict_single(self, image_path):
        """Predict on a single image"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        return prediction.cpu().numpy()[0]
    
    def predict_batch(self, image_paths):
        """Predict on a batch of images"""
        predictions = []
        
        for image_path in tqdm(image_paths, desc="Making predictions"):
            pred = self.predict_single(image_path)
            predictions.append(pred)
        
        return np.array(predictions)

def create_submission(predictor, test_df, output_path='submission.csv'):
    """Create submission file for test data"""
    
    predictions = []
    series_ids = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test data"):
        series_id = row['SeriesInstanceUID']
        series_ids.append(series_id)
        
        # Find images for this series
        image_paths = []
        for artery_dir in os.listdir(config.DATA_DIR):
            artery_path = os.path.join(config.DATA_DIR, artery_dir)
            if os.path.isdir(artery_path):
                series_path = os.path.join(artery_path, series_id)
                if os.path.exists(series_path):
                    png_files = [f for f in os.listdir(series_path) if f.endswith('.png')]
                    for png_file in png_files[:10]:  # Use first 10 images
                        image_paths.append(os.path.join(series_path, png_file))
                    break
        
        if image_paths:
            # Average predictions from multiple images
            all_preds = []
            for img_path in image_paths:
                pred = predictor.predict_single(img_path)
                all_preds.append(pred)
            
            avg_pred = np.mean(all_preds, axis=0)
        else:
            # No images found, use zeros
            avg_pred = np.zeros(config.NUM_CLASSES)
        
        predictions.append(avg_pred)
    
    # Create submission DataFrame
    artery_cols = get_artery_columns()
    submission_df = pd.DataFrame(predictions, columns=artery_cols)
    submission_df.insert(0, 'SeriesInstanceUID', series_ids)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission_df

# =============================================================================
# CELL 10: MODEL EVALUATION AND ANALYSIS
# =============================================================================

def evaluate_model(model_path, val_loader, device):
    """Evaluate model performance"""
    
    # Load model
    model = BrainAneurysmModel(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate AUC for each class
    artery_cols = get_artery_columns()
    auc_scores = {}
    
    for i, col in enumerate(artery_cols):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            auc_scores[col] = auc
        except:
            auc_scores[col] = 0.5
    
    # Plot AUC scores
    plt.figure(figsize=(15, 6))
    classes = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    plt.bar(range(len(classes)), scores)
    plt.title('AUC Scores by Artery Location')
    plt.xlabel('Artery Location')
    plt.ylabel('AUC Score')
    plt.xticks(range(len(classes)), [col.split()[-1] for col in classes], rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean AUC: {np.mean(scores):.4f}")
    print(f"Std AUC: {np.std(scores):.4f}")
    
    return auc_scores

def analyze_predictions(predictions, targets, artery_cols):
    """Analyze model predictions"""
    
    # Calculate confusion matrix for each class
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(artery_cols):
        if i < len(axes):
            # Convert to binary predictions
            pred_binary = (predictions[:, i] > 0.5).astype(int)
            target_binary = targets[:, i].astype(int)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(target_binary, pred_binary)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{col.split()[-1]}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# CELL 11: MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("Starting Brain Aneurysm Detection Training")
    print("="*50)
    
    # Option 1: Run hyperparameter optimization
    # study = optimize_hyperparameters()
    
    # Option 2: Train with cross-validation
    fold_scores = train_with_cross_validation()
    
    print("\nTraining completed!")
    print(f"Average CV Score: {np.mean(fold_scores):.4f}")

# =============================================================================
# CELL 12: USAGE EXAMPLES
# =============================================================================

"""
# Example 1: Quick training with default parameters
# Run this cell to start training with cross-validation
fold_scores = train_with_cross_validation()
print(f"Cross-validation results: {fold_scores}")
print(f"Mean CV AUC: {np.mean(fold_scores):.4f}")

# Example 2: Hyperparameter optimization
# Uncomment to run hyperparameter optimization
# study = optimize_hyperparameters()
# print(f"Best parameters: {study.best_trial.params}")

# Example 3: Make predictions on test data
# Load test data
test_df = pd.read_csv(config.TEST_CSV)

# Initialize predictor with trained model
predictor = Predictor('best_model.pth', device)

# Create submission
submission_df = create_submission(predictor, test_df, 'submission.csv')
print("Submission created successfully!")

# Example 4: Evaluate model performance
# Load validation data
train_df = load_data()
artery_cols = get_artery_columns()
train_df['any_aneurysm'] = train_df[artery_cols].sum(axis=1) > 0

# Split data
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['any_aneurysm'])

# Create validation dataset
_, val_transform = get_transforms()
val_dataset = BrainAneurysmDataset(val_data, config.DATA_DIR, transform=val_transform, is_train=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

# Evaluate model
auc_scores = evaluate_model('best_model.pth', val_loader, device)
print(f"Model evaluation completed!")

# Example 5: Analyze predictions
# Get predictions and targets
all_preds = []
all_targets = []

model = BrainAneurysmModel(num_classes=config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        targets = batch['labels'].to(device)
        
        outputs = model(images)
        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()

# Analyze predictions
analyze_predictions(all_preds, all_targets, artery_cols)
"""
