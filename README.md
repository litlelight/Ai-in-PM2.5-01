# PM-STGformer: Dual-Stage Attention Graph Transformer for PM2.5 Prediction with Chemical Interaction and Uncertainty Modeling

## 🚀 Overview

**PM-STGformer** is a novel spatiotemporal graph Transformer model that revolutionizes PM2.5 concentration prediction by explicitly modeling chemical interactions between pollutants and providing uncertainty quantification. This repository contains the complete implementation of our paper **"Dual-Stage Attention Graph Transformer for PM2.5 Prediction with Chemical Interaction and Uncertainty Modeling"**.

### 🏆 Key Achievements
- **8.7%, 7.5%, and 5.2%** improvements in RMSE, MAE, and R² metrics over state-of-the-art methods
- **>85%** prediction accuracy maintained during severe pollution episodes (PM2.5 > 250 μg/m³)
- **89% and 94%** coverage rates at 90% and 95% confidence levels for uncertainty estimation
- **80.6%** average performance retention in cross-city zero-shot transfer experiments

## 🧠 Technical Innovation

### Dual-Stage Attention Mechanism
- **Pollutant Association Module (PAM)**: Explicitly models complex chemical interactions between atmospheric pollutants through enhanced multi-head attention
- **Spatiotemporal Feature Fusion Module (STFM)**: Captures multi-scale temporal dependencies from hourly variations to seasonal trends

### Probabilistic Prediction Framework
- First PM2.5 prediction model to provide both point estimates and uncertainty quantification
- Gaussian negative log-likelihood loss for joint optimization of accuracy and uncertainty calibration
- Enables risk-based decision making for air quality management

### Graph-Enhanced Architecture
- Dynamic heterogeneous graph construction for pollutant interaction networks
- Multi-scale graph convolution with K-hop message passing
- Integration of atmospheric chemistry prior knowledge

## 📊 Model Architecture

```
Input Features → Feature Embedding → PAM → STFM → Multi-Scale Extraction → Graph Construction → Probabilistic Prediction Head
     ↓                 ↓              ↓      ↓           ↓                    ↓                    ↓
16 Features      256-dim Hidden   Chemical  Temporal   {3,5,7} Kernels    Dynamic Graph      Mean ± Uncertainty
(6 Pollutants +   Representation  Interaction Attention  Multi-Scale       Convolution        PM2.5 Prediction
10 Meteorological)                Modeling   Mechanism   Features         Operations
```

## 🔧 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 2.0.0
NumPy >= 1.24.0
Pandas >= 2.0.0
Scikit-learn >= 1.3.0
```

### Setup
```bash
git clone https://github.com/litlelight/Ai-in-PM2.5-01.git
cd Ai-in-PM2.5-01
pip install -r requirements.txt
```

## 📁 Repository Structure

```
PM-STGformer/
├── src/
│   ├── models/
│   │   ├── pm_stgformer.py          # Main model architecture
│   │   ├── pollutant_association.py # PAM module implementation
│   │   ├── spatiotemporal_fusion.py # STFM module implementation
│   │   └── graph_constructor.py     # Dynamic graph construction
│   ├── data/
│   │   ├── preprocessor.py          # Data preprocessing pipeline
│   │   ├── dataset.py               # PyTorch dataset implementation
│   │   └── feature_engineering.py   # Feature extraction utilities
│   ├── training/
│   │   ├── trainer.py               # Training loop implementation
│   │   ├── losses.py                # Probabilistic loss functions
│   │   └── metrics.py               # Evaluation metrics
│   └── utils/
│       ├── visualization.py         # Result visualization tools
│       └── uncertainty_analysis.py  # Uncertainty calibration tools
├── data/
│   ├── beijing_aqi_2013_2017.csv    # Beijing air quality dataset
│   ├── cross_city_data/             # Shanghai, Guangzhou, Chengdu data
│   └── preprocessed/                # Processed datasets
├── experiments/
│   ├── configs/                     # Experiment configurations
│   ├── results/                     # Experimental results
│   └── notebooks/                   # Analysis notebooks
├── models/
│   ├── pretrained/                  # Pre-trained model weights
│   └── checkpoints/                 # Training checkpoints
└── docs/
    ├── API_documentation.md         # Detailed API documentation
    ├── experiment_reproduction.md   # Steps to reproduce results
    └── model_architecture.md        # Detailed architecture explanation
```

## 🚀 Quick Start

### 1. Data Preparation
```bash
python src/data/preprocessor.py --dataset beijing --output_dir data/preprocessed/
```

### 2. Model Training
```bash
python src/training/trainer.py --config experiments/configs/beijing_config.yaml
```

### 3. Evaluation
```bash
python evaluate.py --model_path models/pretrained/pm_stgformer_beijing.pth --test_data data/preprocessed/beijing_test.pkl
```

### 4. Inference with Uncertainty
```python
from src.models.pm_stgformer import PMSTGformer
import torch

# Load pre-trained model
model = PMSTGformer.load_from_checkpoint('models/pretrained/pm_stgformer_beijing.pth')
model.eval()

# Make predictions with uncertainty
with torch.no_grad():
    mean_pred, uncertainty = model(input_data)
    confidence_interval = (mean_pred - 1.96 * uncertainty, mean_pred + 1.96 * uncertainty)
```

## 📈 Experimental Results

### Comparative Performance (Beijing Agricultural Exhibition Hall Station)

| Method | RMSE (μg/m³) | MAE (μg/m³) | MAPE (%) | R² |
|--------|--------------|-------------|----------|-----|
| Linear Regression | 48.23 | 35.84 | 32.15 | 0.623 |
| XGBoost | 32.84 | 24.23 | 21.67 | 0.741 |
| LSTM | 29.45 | 21.38 | 19.82 | 0.778 |
| CNN-LSTM | 27.16 | 19.73 | 18.45 | 0.801 |
| AirFormer | 26.18 | 18.92 | 17.83 | 0.815 |
| ST-iTransformer | 24.92 | 17.84 | 16.81 | 0.832 |
| **PM-STGformer** | **23.60** | **16.82** | **15.23** | **0.847** |

### Uncertainty Calibration Results
- **90% Confidence Level**: 89.0% actual coverage (ideal: 90%)
- **95% Confidence Level**: 94.0% actual coverage (ideal: 95%)
- **Reliability Index**: 0.945 (closer to 1.0 indicates better calibration)

### Cross-City Generalization
| City | Zero-shot RMSE | 20% Fine-tuned RMSE | Retention Rate |
|------|---------------|---------------------|----------------|
| Shanghai | 31.20 | 25.92 | 81.4% |
| Guangzhou | 28.80 | 22.95 | 76.7% |
| Chengdu | 34.50 | 29.03 | 83.8% |

## 🎯 Ablation Study Results

| Component Removed | RMSE (μg/m³) | Performance Drop |
|-------------------|--------------|------------------|
| Full Model | 23.60 | - |
| w/o STFM | 26.20 | +11.0% |
| w/o PAM | 25.80 | +9.3% |
| w/o Multi-scale | 25.11 | +6.4% |
| w/o Uncertainty | 24.31 | +3.0% |

## 📊 Dataset Information

### Beijing Air Quality Dataset
- **Time Period**: March 2013 - February 2017
- **Frequency**: Hourly measurements
- **Total Records**: 35,063 observations
- **Features**: 16 variables (6 pollutants + 10 meteorological)
- **Missing Data**: 2.3% (handled via temporal interpolation)

### Pollutants
- PM2.5, PM10, SO₂, NO₂, O₃, CO

### Meteorological Variables
- Temperature (TEMP), Pressure (PRES), Dew Point (DEWP)
- Precipitation (RAIN), Wind Direction (wd), Wind Speed (WSPM)

## 🔬 Reproducibility

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **RAM**: 128GB recommended for full dataset processing
- **Storage**: 50GB available space

### Training Configuration
- **Batch Size**: 512
- **Learning Rate**: 1e-3 with cosine annealing
- **Epochs**: 200 with early stopping
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Training Time**: ~6 hours on dual RTX 4090

### Reproducing Results
```bash
# Run complete experimental pipeline
bash scripts/reproduce_all_experiments.sh

# Individual experiments
python experiments/run_comparative_study.py
python experiments/run_ablation_study.py
python experiments/run_cross_city_validation.py
python experiments/run_uncertainty_analysis.py
```

## 📋 Model Configuration

### Optimal Hyperparameters (from ablation study)
```yaml
model:
  hidden_dim: 256
  num_heads: 4
  seq_len: 36  # hours
  dropout: 0.1
  kernel_sizes: [3, 5, 7]

training:
  loss_alpha: 0.5  # MSE-NLL balance
  learning_rate: 1e-3
  weight_decay: 1e-5
  gradient_clip: 1.0
```

## 🎨 Visualization Tools

The repository includes comprehensive visualization utilities:

```bash
# Generate attention weight visualizations
python src/utils/visualization.py --type attention_weights --model_path models/pretrained/

# Plot prediction vs. ground truth with uncertainty bands
python src/utils/visualization.py --type prediction_analysis --test_data data/test/

# Uncertainty calibration plots
python src/utils/uncertainty_analysis.py --generate_plots
```

## 📚 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{yang2024pmstgformer,
  title={Dual-Stage Attention Graph Transformer for PM2.5 Prediction with Chemical Interaction and Uncertainty Modeling},
  author={Yang, YanCheng and Zhang, YuChen},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Beijing Environmental Monitoring Center for providing air quality data
- National Meteorological Information Center for meteorological data
- PyTorch and Hugging Face communities for excellent deep learning frameworks

## 📞 Contact

For questions about the code or paper, please contact:
- **YanCheng Yang**: [1192448328@qq.com](mailto:1192448328@qq.com)
- **YuChen Zhang**: [2627556529@qq.com](mailto:2627556529@qq.com) (Corresponding Author)

## 🔄 Updates

- **v1.0.0** (2024-01): Initial release with complete PM-STGformer implementation
- **v1.1.0** (2024-02): Added cross-city validation experiments
- **v1.2.0** (2024-03): Enhanced uncertainty analysis tools

---

**Note**: This implementation represents the core research contribution. Production deployment may require additional optimization and calibration specific to local conditions and regulatory requirements.
