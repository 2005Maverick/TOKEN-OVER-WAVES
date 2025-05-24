import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.positional_encoding import PositionalEncoding
from models.multi_resolution_tokenizer import MultiResolutionTokenizer

class AudioClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        
        # Feature extraction
        self.tokenizer = MultiResolutionTokenizer(
            input_dim=1,
            embedding_dim=embedding_dim,
            kernel_sizes=[3, 7, 15],
            strides=[1, 2, 4]
        )
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_rate
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, mel_spec):
        # Process mel spectrogram
        x = mel_spec.squeeze(1)  # Remove channel dimension
        x = self.tokenizer(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer processing
        x = self.transformer_encoder(x)
        
        # Pooling
        x = x.mean(dim=1)  # Mean pooling
        
        # Classification
        x = self.classifier(x)
        return x.squeeze(1)

class AudioClassifierPL(pl.LightningModule):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.model = AudioClassifier(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, mel_spec):
        return self.model(mel_spec)
    
    def training_step(self, batch, batch_idx):
        mel_spec = batch['mel_spec']
        labels = batch['label'].float()
        
        outputs = self(mel_spec)
        loss = self.criterion(outputs, labels)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mel_spec = batch['mel_spec']
        labels = batch['label'].float()
        
        outputs = self(mel_spec)
        loss = self.criterion(outputs, labels)
        
        # Calculate metrics
        preds = torch.sigmoid(outputs) > 0.5
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-5,
            weight_decay=0.01
        )
        return optimizer
