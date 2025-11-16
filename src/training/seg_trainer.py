"""
Trainer for 3D segmentation models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, List
import logging
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from segmentation.unet_3d import UNet3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, D, H, W)
        """
        # Apply softmax to predictions
        pred = torch.softmax(pred, dim=1)

        # Convert target to one-hot
        num_classes = pred.shape[1]
        target_one_hot = torch.nn.functional.one_hot(
            target,
            num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # Calculate Dice score
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """Combined Cross Entropy and Dice Loss"""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class SegmentationTrainer:
    """
    Trainer for 3D segmentation models
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        output_dir: str = 'output/training',
        checkpoint_interval: int = 10,
        early_stopping_patience: int = 15
    ):
        """
        Initialize Segmentation Trainer

        Args:
            model: Segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Number of epochs
            output_dir: Directory to save outputs
            checkpoint_interval: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping_patience = early_stopping_patience

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info(f"Trainer initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate Dice score
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                dice = self.calculate_dice(pred, labels)

            total_loss += loss.item()
            total_dice += dice
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return {'loss': avg_loss, 'dice': avg_dice}

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Calculate Dice score
                pred = torch.argmax(outputs, dim=1)
                dice = self.calculate_dice(pred, labels)

                total_loss += loss.item()
                total_dice += dice
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}'
                })

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return {'loss': avg_loss, 'dice': avg_dice}

    def calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient"""
        smooth = 1.0
        # Assuming binary segmentation (background vs foreground)
        pred_binary = (pred > 0).float()
        target_binary = (target > 0).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def save_history(self):
        """Save training history"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        start_time = datetime.now()

        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                logger.info("✓ New best model!")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            if epoch % self.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Save history
            self.save_history()

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                logger.info(f"No improvement for {self.early_stopping_patience} epochs")
                break

        # Training complete
        duration = datetime.now() - start_time
        logger.info(f"\nTraining completed in {duration}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history


def main():
    """Example training script"""
    import argparse

    parser = argparse.ArgumentParser(description='Train 3D Segmentation Model')
    parser.add_argument('--train-dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--output-dir', type=str, default='output/training', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Import data loaders
    from data_loaders import create_segmentation_dataloaders

    # Create data loaders
    train_loader, val_loader = create_segmentation_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size
    )

    # Create model
    model = UNet3D(
        in_channels=1,
        num_classes=args.num_classes,
        base_features=32
    )

    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        output_dir=args.output_dir
    )

    # Train
    history = trainer.train()

    print("\nTraining completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
