# BrainPath: Predicting Subject-Specific Brain Aging Trajectories

## Overview

BrainPath is a U-Net-like encoder-decoder framework designed for subject-specific MRI prediction at arbitrary timepoints. We chose a U-Net as the base architecture over diffusion or GAN-based models because it allows the fine-grained anatomical features of the input MRI to be directly propagated to the output. This is a crucial advantage for our task, as brain aging is a gradual process characterized by subtle structural changes over time.

## Key Innovations

Building on the U-Net architecture, BrainPath introduces several innovative modifications to its architecture, loss function, and training strategy:

### 1. Age-Conditioned Architecture

The encoder is augmented with an **age regression head** to learn age-related representations. This head predicts the brain age of both the input and target images during training, and the predicted age difference is passed to the decoder as a conditioning input. This conditioning enables the decoder to synthesize MRIs that reflect biologically meaningful age-related changes.

### 2. Biologically-Informed Loss Functions

The overall loss function incorporates a reconstruction loss along with two biologically-informed losses:

- **Age Calibration Loss**: Ensures that the predicted brain age difference between two scans from the same subject matches their chronological age difference and that the group-level mean of predicted brain ages aligns with the mean chronological age.

- **Age Perceptual Loss**: Compares the feature-level representations and predicted age from the reconstructed and actual MRIs. This loss emphasizes subtle, temporally meaningful structural variations that are often obscured by imaging artifacts or suboptimal image preprocessing.

### 3. Novel Swap Learning Strategy

We introduce a novel swap learning strategy to enhance robustness and promote implicit disentanglement between subject-specific anatomy and age-related changes. During training, two MRIs from the same subject at different timepoints are randomly assigned as input and target, with their roles swapped in subsequent passes. This strategy removes the need for the encoder to explicitly disentangle "structure" and "age" channels, while allowing the decoder to condition solely on the regressed brain age difference, free from the bias of raw chronological labels.

## Technical Features

### Network Architecture
- **3D U-Net** based on ResNet blocks
- **Age regression head** integrated into the encoder
- **Feature separation design**: structural features and age features are processed independently
- **Multi-scale feature fusion** throughout the encoder-decoder pathway

### Data Processing
- Supports multi-scale processing from **256³ to 120³** resolution
- **3D data augmentation**: rotation, flipping, random cropping
- **Patch-level mixing** augmentation strategy
- **Swap learning** for implicit disentanglement

### Training Strategy
- **Two-stage training**: Stage 0 initialization, Stage 1 fine-tuning
- **Multiple loss functions**: reconstruction loss, age calibration loss, age perceptual loss
- **Gradient clipping** to prevent gradient explosion
- **Learning rate scheduling** and early stopping mechanism

## Project Structure

```
Brain_path_share/
├── ae_3d.py              # 3D U-Net encoder and decoder definitions
├── unet_3d_main.py       # Main training script with swap learning
├── utils.py              # Data loading and processing utilities
├── 3dunet/               # Experimental results directory
│   ├── ex_2025_02_26_08-43-39/  # First part training with all checkpoints and visualizations
│   └── ex_2025_03_01_10-19-02/  # Second part training (resumed from first part)
└── README.md             # Project documentation
```

## Core Components

### 1. 3D U-Net Architecture (`ae_3d.py`)

- **UNetEncoder3D**: 3D encoder with integrated age regression head that extracts spatial features and predicts brain age
- **UNetDecoder3D**: 3D decoder that reconstructs images conditioned on features and age difference
- **UNetModel3D**: Complete encoder-decoder model with age conditioning

### 2. Data Processing (`utils.py`)

- **MRIDataset_3D_three**: 3D MRI dataset loader with swap learning support
- **Data Augmentation**: Comprehensive 3D augmentation pipeline
- **Patch Mixing**: Advanced patch-level data mixing for robustness
- **Swap Learning**: Implementation of the novel training strategy

### 3. Training Pipeline (`unet_3d_main.py`)

- **Biologically-informed losses**: Age calibration and perceptual losses
- **Swap learning implementation**: Dynamic input-target role swapping
- **Comprehensive visualization**: t-SNE feature visualization, age prediction analysis

## Requirements

```python
torch >= 1.8.0
torchvision
numpy
scikit-learn
matplotlib
opencv-python
wandb
tqdm
nibabel
scipy
ruptures
```

## Usage

### 1. Path Configuration

Before training, configure the base path (your working directory) in `unet_3d_main.py` (line 27):
```python
BASE_PATH = "../../../pub/liyifan/mri"  # Modify to your data path
```
If you are loading a pretrained model weight, config the path_model in  `unet_3d_main.py` (line 606):
```python
path_model = BASE_PATH + "/3dunet/ex_2025_02_26_08-43-39/check_point_model_290_1_1.pth" #replace this with the path of your pretrained model
```
### 2. Data Preparation
The source data are available via requests to the respective databases. https://adni.loni.usc.edu/ (ADNI) and https://naccdata.org/ (NACC). And we will also release the preprocessed data at the link: https://zenodo.org/records/17237876.

Organize MRI data with subject-specific timepoint information (If you download the zip files in the zenodo website provided above it has already been sturctured):

**Dataset Directory Structure:**
```
{BASE_PATH}/FS_2/
├── processed_health_all/
│   ├── train_val/
│   │   ├── sample_rid1_j1_age1/
│   │   │   └── 3d_mni.pkl
│   │   ├── sample_rid1_j2_age2/
│   │   │   └── 3d_mni.pkl
│   │   └── ...
│   ├── validation/
│   │   └── ...
│   └── test/
│       └── ...
```

**Sample Folder Naming:** `sample_{rid}_{j}_{age}/`
- `rid`: Subject unique ID
- `j`: Timepoint index  
- `age`: Age at this timepoint

### 3. Model Save Path

Models and outputs are automatically saved to:
```
{BASE_PATH}/3dunet/ex_{YYYY_MM_DD_HH-MM-SS}/
├── check_point_model_{epoch}_1_1.pth  # Model checkpoints (every 10 epochs)
├── image/                              # Training/test visualizations
├── logs.txt                            # Training logs
├── unet_3d_main.py                     # Source code backup
├── ae_3d.py
└── utils.py
```

### 4. Training
```bash
# Basic training with default parameters
python unet_3d_main.py

# Training with custom parameters
python unet_3d_main.py --batch_size 24 --lr 1e-5 --epochs 300
```

### 5. Key Parameters
- `--batch_size`: Batch size (default: 24)
- `--lr`: Learning rate (default: 1e-5)
- `--epochs`: Number of training epochs (default: 300)
- `--resize_dim`: Image dimensions (default: 128)
- `--structure_vec_size`: Structure vector size (default: 256)
- `--longitudinal_vec_size`: Longitudinal vector size (default: 256)

## Experimental Results

The project demonstrates the effectiveness of the swap learning strategy and biologically-informed losses:

- **Age Prediction Accuracy**: High correlation between predicted and chronological age differences
- **Structural Preservation**: Fine-grained anatomical features maintained across timepoints
- **Biological Plausibility**: Generated changes consistent with known aging patterns

## Visualization Features

- **Age Prediction Analysis**: Scatter plots showing predicted vs actual age relationships
- **Feature Space Visualization**: t-SNE plots of learned representations
- **Training Progress**: Input, reconstructed, and target image comparisons
- **Longitudinal Changes**: Visualization of predicted aging trajectories

## Method Advantages

1. **Biological Grounding**: Age-aware architecture and losses ensure meaningful predictions
2. **Implicit Disentanglement**: Swap learning removes need for explicit structure/age separation
3. **Fine-grained Control**: Precise conditioning on age differences for targeted predictions
4. **Robustness**: Comprehensive augmentation and swap strategy enhance generalization
5. **Interpretability**: Clear separation of anatomical preservation and aging effects

## Citation

If you use BrainPath in your research, please cite:

```bibtex
@misc{brainpath_2025,
  title={BrainPath: Predicting Subject-Specific Brain Aging Trajectories},
  author={Yifan Li, Javad Sohankar, Ji Luo, Jing Li*, Yi Su*},
  year={2025},
  url={https://github.com/formanuscriptsharing/Brain_path}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please contact us through GitHub Issues. 