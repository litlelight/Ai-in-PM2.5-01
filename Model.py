import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PollutantAssociationModule(nn.Module):
    """
    Pollutant Association Module (PAM) - Core component for chemical interaction modeling
    """

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super(PollutantAssociationModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention for pollutant interactions
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Chemical interaction enhancement (key detail - needs fine-tuning)
        self.chemical_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, pollutant_mask=None):
        """
        x: [batch_size, seq_len, num_pollutants, hidden_dim]
        """
        batch_size, seq_len, num_pollutants, hidden_dim = x.shape

        # Reshape for attention computation
        x_reshaped = x.view(batch_size * seq_len, num_pollutants, hidden_dim)

        # Apply multi-head attention across pollutants
        attn_output, attn_weights = self.multihead_attn(
            x_reshaped, x_reshaped, x_reshaped,
            key_padding_mask=pollutant_mask
        )

        # Chemical interaction enhancement (proprietary weighting - needs tuning)
        enhanced = self.chemical_enhancer(attn_output)

        # Residual connection and normalization
        output = self.norm1(x_reshaped + self.dropout(enhanced))

        # Reshape back
        output = output.view(batch_size, seq_len, num_pollutants, hidden_dim)

        return output, attn_weights


class SpatioTemporalFeatureFusionModule(nn.Module):
    """
    Spatiotemporal Feature Fusion Module (STFM) - Handles multi-scale temporal dependencies
    """

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super(SpatioTemporalFeatureFusionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Temporal attention mechanism
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = TransformerEncoder(encoder_layer, num_layers=2)

        # Position encoding for temporal information
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Fusion mechanism (critical component - requires careful tuning)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_pollutants, hidden_dim]
        """
        batch_size, seq_len, num_pollutants, hidden_dim = x.shape

        # Reshape for temporal processing
        x_temporal = x.permute(0, 2, 1, 3).contiguous()  # [batch, pollutants, seq_len, hidden]
        x_temporal = x_temporal.view(batch_size * num_pollutants, seq_len, hidden_dim)

        # Add positional encoding
        x_temporal = self.pos_encoding(x_temporal)

        # Apply temporal transformer
        temporal_features = self.temporal_encoder(x_temporal)

        # Apply fusion mechanism
        fused_features = self.temporal_fusion(temporal_features)

        # Reshape back
        output = fused_features.view(batch_size, num_pollutants, seq_len, hidden_dim)
        output = output.permute(0, 2, 1, 3).contiguous()

        return output


class MultiScaleFeatureExtraction(nn.Module):
    """
    Multi-scale temporal convolution module
    """

    def __init__(self, hidden_dim=256, kernel_sizes=[3, 5, 7]):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.kernel_sizes = kernel_sizes

        # Multi-scale convolution branches
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(k, 1),
                      padding=(k // 2, 0), groups=hidden_dim // 4)  # Note: groups parameter is critical
            for k in kernel_sizes
        ])

        # Feature fusion
        self.fusion = nn.Conv2d(hidden_dim * len(kernel_sizes), hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_pollutants, hidden_dim]
        """
        # Reshape for conv2d: [batch, hidden_dim, seq_len, num_pollutants]
        x_conv = x.permute(0, 3, 1, 2).contiguous()

        # Apply multi-scale convolutions
        multi_scale_features = []
        for conv in self.conv_branches:
            features = conv(x_conv)
            multi_scale_features.append(features)

        # Concatenate and fuse
        concatenated = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion(concatenated)

        # Reshape back and normalize
        output = fused.permute(0, 2, 3, 1).contiguous()
        output = self.norm(output)
        output = self.activation(output)

        return output


class GraphStructureConstructor(nn.Module):
    """
    Dynamic graph construction for pollutant interactions
    """

    def __init__(self, num_pollutants=6, hidden_dim=256):
        super(GraphStructureConstructor, self).__init__()
        self.num_pollutants = num_pollutants

        # Learnable node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_pollutants, hidden_dim))

        # Prior adjacency matrix (chemical knowledge - needs domain expertise)
        prior_adj = self._create_prior_adjacency()
        self.register_buffer('prior_adj', prior_adj)

        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)  # 3-hop message passing
        ])

    def _create_prior_adjacency(self):
        """
        Create prior adjacency matrix based on chemical knowledge
        Note: This is simplified - actual chemical relationships need expert knowledge
        """
        prior = torch.eye(self.num_pollutants)
        # Add some basic chemical relationships (simplified)
        # TODO: This needs proper atmospheric chemistry knowledge
        prior[0, 1] = 0.8  # PM2.5 <-> PM10
        prior[1, 0] = 0.6
        prior[0, 2] = 0.7  # PM2.5 <-> SO2
        prior[2, 0] = 0.7
        # ... more relationships need to be defined
        return prior

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_pollutants, hidden_dim]
        """
        batch_size, seq_len, num_pollutants, hidden_dim = x.shape

        # Compute adaptive adjacency matrix
        embeddings = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        adj_matrix = torch.softmax(
            F.relu(torch.bmm(embeddings, embeddings.transpose(1, 2))) + self.prior_adj,
            dim=-1
        )

        # Apply graph convolutions
        graph_features = x
        for i, conv in enumerate(self.graph_convs):
            # Multi-hop message passing (implementation detail varies)
            adj_power = torch.matrix_power(adj_matrix, i + 1)
            messages = torch.einsum('bnij,btjd->btid', adj_power, graph_features)
            graph_features = graph_features + conv(messages)

        return graph_features


class ProbabilisticPredictionHead(nn.Module):
    """
    Dual-branch prediction head for mean and uncertainty estimation
    """

    def __init__(self, hidden_dim=256, output_dim=1):
        super(ProbabilisticPredictionHead, self).__init__()

        # Mean prediction branch
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Uncertainty estimation branch
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_pollutants, hidden_dim]
        """
        # Focus on PM2.5 (index 0) for prediction
        pm25_features = x[:, -1, 0, :]  # Last timestep, PM2.5

        mean = self.mean_head(pm25_features)
        uncertainty = self.uncertainty_head(pm25_features)

        return mean, uncertainty


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class PMSTGformer(nn.Module):
    """
    Main PM-STGformer model architecture
    """

    def __init__(self,
                 input_dim=16,  # Number of input features
                 hidden_dim=256,  # Hidden dimension
                 num_pollutants=6,  # Number of pollutants
                 seq_len=36,  # Input sequence length
                 num_heads=4,  # Attention heads
                 dropout=0.1):
        super(PMSTGformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_pollutants = num_pollutants

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # Core modules
        self.pam = PollutantAssociationModule(hidden_dim, num_heads, dropout)
        self.stfm = SpatioTemporalFeatureFusionModule(hidden_dim, num_heads, dropout)
        self.multi_scale = MultiScaleFeatureExtraction(hidden_dim)
        self.graph_constructor = GraphStructureConstructor(num_pollutants, hidden_dim)

        # Prediction head
        self.prediction_head = ProbabilisticPredictionHead(hidden_dim)

        # Initialize weights (critical for performance)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Weight initialization - this is crucial but implementation details matter
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with specific gain (needs tuning)
                nn.init.xavier_normal_(module.weight, gain=0.8)  # Note: gain value is critical
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_features]
        """
        batch_size, seq_len, num_features = x.shape

        # Reshape to separate pollutants and meteorological features
        # Note: This assumes specific feature ordering - needs careful data preprocessing
        pollutant_features = x[:, :, :self.num_pollutants]  # First 6 features are pollutants
        met_features = x[:, :, self.num_pollutants:]  # Rest are meteorological

        # Combine features (implementation detail varies)
        combined_features = torch.cat([
            pollutant_features,
            met_features.unsqueeze(2).expand(-1, -1, self.num_pollutants, -1)
        ], dim=-1)

        # Input embedding
        embedded = self.input_embedding(combined_features)
        embedded = self.feature_norm(embedded)

        # Apply core modules
        pam_output, _ = self.pam(embedded)
        stfm_output = self.stfm(pam_output)
        multi_scale_output = self.multi_scale(stfm_output)
        graph_output = self.graph_constructor(multi_scale_output)

        # Prediction
        mean, uncertainty = self.prediction_head(graph_output)

        return mean, uncertainty


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for probabilistic training
    """

    def __init__(self, alpha=0.5):
        super(GaussianNLLLoss, self).__init__()
        self.alpha = alpha  # Balance between MSE and NLL

    def forward(self, mean, uncertainty, target):
        # NLL component
        nll = 0.5 * torch.log(2 * math.pi * uncertainty ** 2) + \
              (target - mean) ** 2 / (2 * uncertainty ** 2)

        # MSE component
        mse = F.mse_loss(mean, target)

        # Combined loss (weight balance is critical)
        total_loss = self.alpha * nll.mean() + (1 - self.alpha) * mse

        return total_loss


# Example usage and training setup
def create_model():
    """
    Factory function to create PM-STGformer model
    """
    model = PMSTGformer(
        input_dim=16,  # 6 pollutants + 10 meteorological features
        hidden_dim=256,  # Note: This hyperparameter is critical
        num_pollutants=6,
        seq_len=36,  # 36-hour input window
        num_heads=4,  # Optimal number from ablation study
        dropout=0.1
    )
    return model


def get_optimizer_and_scheduler(model, lr=1e-3):
    """
    Optimizer and scheduler setup - parameters need fine-tuning
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,  # L2 regularization strength
        betas=(0.9, 0.999)  # Adam parameters
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,  # Total epochs
        eta_min=1e-6
    )

    return optimizer, scheduler

# Note: This implementation provides the core architecture but several critical details
# for optimal performance are left as "implementation details" that require:
# 1. Careful hyperparameter tuning
# 2. Proper data preprocessing and feature engineering
# 3. Training strategy optimization
# 4. Loss function weight balancing
# 5. Chemical knowledge integration in graph construction