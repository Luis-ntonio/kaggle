import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
# --- Inception1D Block for multi-scale convolution ---
class Inception1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0, "out_ch must be divisible by 4"
        branch_ch = out_ch // 4
        self.b1 = nn.Conv1d(in_ch, branch_ch, kernel_size=1, padding=0)
        self.b2 = nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1)
        self.b3 = nn.Conv1d(in_ch, branch_ch, kernel_size=5, padding=2)
        self.b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch_ch, kernel_size=1)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(torch.cat([b(x) for b in (self.b1, self.b2, self.b3, self.b4)], dim=1))

# --- Squeeze-and-Excitation block to reweight channels ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        # x: (B, C, T)
        s = x.mean(-1)  # (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)  # (B, C, 1)
        return x * s

# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (B, T, D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

# --- Transformer Encoder for temporal modeling ---
class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=1024):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
    def forward(self, x, padding_mask):
        # x: (B, T, D); mask: (B, T) float mask 1/0
        x = self.pos_enc(x)
        # transformer expects True to mask
        #key_mask = ~mask.bool()
        return self.encoder(x, src_key_padding_mask=padding_mask)

# --- Attention Pooling along time dimension ---
class AttnPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x, mask):
        # x: (B, T, C), mask: (B, T)
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * alpha).sum(dim=1)

# --- Improved BFRB Classifier combining all ideas ---
class ImprovedBFRBClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 37,
        demog_dim: int = 7,
        inception_channels: List[int] = [64, 128],
        transform_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        lstm_hidden: int = 128,
        drop: float = 0.3,
        num_classes: int = 8
    ):
        super().__init__()
        # Inception blocks
        self.inc1 = Inception1D(input_dim, inception_channels[0])
        self.inc2 = Inception1D(inception_channels[0], inception_channels[1])
        self.se   = SEBlock(inception_channels[1])
        # Project to transformer dim
        self.proj = nn.Linear(inception_channels[1], transform_dim)
        # Transformer
        self.tr   = TemporalTransformer(
            d_model=transform_dim, nhead=transformer_heads,
            num_layers=transformer_layers, dim_feedforward=transform_dim*2,
            dropout=drop, max_len=2048
        )
        # Attention pooling
        self.attn_pool = AttnPool1D(transform_dim)
        # Fusion with demographics
        self.fuse = nn.Sequential(
            nn.Linear(transform_dim + demog_dim, transform_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
        # Heads
        self.target_head = nn.Sequential(
            nn.Linear(transform_dim, transform_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(transform_dim//2, 1)
        )
        self.gesture_head = nn.Sequential(
            nn.Linear(transform_dim, transform_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(transform_dim//2, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, demog: torch.Tensor):
        # x: (B, T, 37), mask: (B, T), demog: (B, 7)
        B, T, _ = x.shape
        if torch.isnan(x).any():
            print("NaNs detected in input features!")
            import pdb; pdb.set_trace()
        # Inception + SE
        z = x.permute(0,2,1)             # (B,37,T)
        if torch.isnan(z).any():
            print("NaNs detected after permute!")
            import pdb; pdb.set_trace()
        z = self.inc1(z)                 # (B,64,T)
        if torch.isnan(z).any():
            print("NaNs detected after inc1!")
            import pdb; pdb.set_trace()
        z = self.inc2(z)                 # (B,128,T)
        if torch.isnan(z).any():
            print("NaNs detected after inc2!")
            import pdb; pdb.set_trace()
        z = self.se(z)                   # (B,128,T)
        if torch.isnan(z).any():
            print("NaNs detected after SE block!")
            import pdb; pdb.set_trace()
        # back to (B,T,128)
        z = z.permute(0,2,1)
        if torch.isnan(z).any():
            print("NaNs detected after permute back!")
            import pdb; pdb.set_trace()
        # project
        z = self.proj(z)                 # (B,T,transform_dim)
        if torch.isnan(z).any():
            print("NaNs detected after projection!")
            import pdb; pdb.set_trace()
        # transformer
        z = self.tr(z, mask)             # (B,T,transform_dim)
        if torch.isnan(z).any():
            print("NaNs detected after transformer!")
            import pdb; pdb.set_trace()
        # attention pooling
        z = self.attn_pool(z, mask)      # (B,transform_dim)
        if torch.isnan(z).any():
            print("NaNs detected after attention pooling!")
            import pdb; pdb.set_trace()
        # fuse demographics
        z = self.fuse(torch.cat([z, demog], dim=1))  # (B,transform_dim)
        if torch.isnan(z).any():
            print("NaNs detected after demographic fusion!")
            import pdb; pdb.set_trace()
        # heads
        lt = self.target_head(z).squeeze(1)
        if torch.isnan(lt).any():
            print("NaNs detected in target head output!")
            import pdb; pdb.set_trace()
        lg = self.gesture_head(z)
        if torch.isnan(lg).any():
            print("NaNs detected in gesture head output!")
            import pdb; pdb.set_trace()
        return lt, lg
