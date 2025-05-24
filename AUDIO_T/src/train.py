import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils.config import Config
from utils.data import DeepfakeAudioDataset, AudioTransforms
from models.audio_classifier import AudioClassifier

def main():
    # Load configuration
    config = Config()
    
    # Set up transforms and datasets
    transforms = AudioTransforms(
        sample_rate=config.SAMPLE_RATE,
        max_length=config.MAX_AUDIO_LENGTH,
        n_fft=config.N_FFT,
        n_mels=config.N_MELS,
        hop_length=config.HOP_LENGTH,
        training=True
    )
    
    train_dataset = DeepfakeAudioDataset(
        root_dir=config.DATA_DIR,
        split='training',
        transform=transforms
    )
    
    val_dataset = DeepfakeAudioDataset(
        root_dir=config.DATA_DIR,
        split='validation',
        transform=transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    
    # Initialize model
    model = AudioClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/',
        filename='audio_classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=TensorBoardLogger('logs/', name='audio_classifier'),
        progress_bar_refresh_rate=20
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
