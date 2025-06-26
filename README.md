# FA2

## Environment Setup

### Option 1: Using Conda (Recommended)

1. Create and activate a new conda environment:
```bash
conda env create -f environment.yml
conda activate FA2
```

2. Verify the installation:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

### Option 2: Manual Installation

If you prefer to set up the environment manually, you can create a new conda environment and install the dependencies:

1. Create and activate a new conda environment:
```bash
conda create -n FA2 python=3.8
conda activate FA2
```

2. Install PyTorch and CUDA dependencies:
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
```

3. Install other required packages:
```bash
pip install monai==1.0.0 nibabel==5.2.1 opencv-python==4.11.0.86
pip install scikit-image==0.21.0 wandb==0.20.1 h5py==3.11.0
pip install matplotlib scipy tqdm
```

### Hardware Requirements

- NVIDIA GPU with CUDA support (CUDA 11.2 or higher)
- RAM: 16GB or more recommended
- Storage: Depends on your dataset size

### Notes

- Make sure your NVIDIA drivers are up to date
- The code has been tested with Python 3.8 and PyTorch 1.10.1
- For training on multiple GPUs, ensure that CUDA toolkit and drivers are properly installed
