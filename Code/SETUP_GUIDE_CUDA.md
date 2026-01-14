# CUDA Setup Guide for DL Evaluation

This guide explains how to set up CUDA and cuDNN for GPU-accelerated PyTorch on Windows with Python 3.14.

## Prerequisites

- **NVIDIA GPU** (CUDA Compute Capability 3.0 or higher)
- **NVIDIA Driver** (latest version recommended)
- **Python 3.14** (or your chosen version)
- **pip** (latest version)

## Quick Setup

### Step 1: Upgrade pip, setuptools, and wheel

```powershell
py -m pip install --upgrade pip setuptools wheel
```

### Step 2: Install NVIDIA cuDNN (CUDA runtime libraries)

This installs NVIDIA cuDNN, which provides GPU acceleration libraries for CUDA 12.1:

```powershell
py -m pip install nvidia-cudnn-cu12 --extra-index-url https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install PyTorch with CUDA Support

Reinstall PyTorch with CUDA 12.1 support:

```powershell
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Verify CUDA Installation

Test that CUDA is properly configured:

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output (if GPU detected):
```
CUDA Available: True
GPU Device: NVIDIA GeForce RTX 3090 (or your GPU name)
```

## Troubleshooting

### CUDA Not Detected

If `torch.cuda.is_available()` returns `False`:

1. **Check NVIDIA Driver:**
   ```powershell
   nvidia-smi
   ```
   If this command fails, update your NVIDIA driver from https://www.nvidia.com/Download/driverDetails.aspx

2. **Reinstall PyTorch:**
   ```powershell
   py -m pip uninstall torch torchvision torchaudio -y
   py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Check CUDA Toolkit Installation:**
   ```powershell
   py -c "import torch; print(torch.version.cuda)"
   ```

### cuDNN Errors

If you see cuDNN-related errors:

1. **Reinstall cuDNN:**
   ```powershell
   py -m pip uninstall nvidia-cudnn-cu12 -y
   py -m pip install nvidia-cudnn-cu12 --extra-index-url https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Check cuDNN version:**
   ```powershell
   python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version()}')"
   ```

## Using the Notebooks

### CPU-Only Notebook

Use this if you don't have CUDA or prefer CPU processing:

```
dl_eval_notebook.ipynb
```

**Activation:**
```powershell
. .\.venv\Scripts\Activate.ps1
jupyter notebook dl_eval_notebook.ipynb
```

### GPU/CUDA-Enabled Notebook

Use this for GPU acceleration:

```
dl_eval_notebook_cuda.ipynb
```

**Activation:**
```powershell
. .\.venv\Scripts\Activate.ps1
jupyter notebook dl_eval_notebook_cuda.ipynb
```

**Features:**
- Automatic CUDA detection
- GPU memory optimization
- Larger batch sizes for GPU processing
- GPU memory monitoring during execution
- Automatic fallback to CPU if CUDA unavailable

## Performance Comparison

Expected speedup with GPU (NVIDIA RTX 3090, as example):

| Task | CPU | GPU | Speedup |
|------|-----|-----|---------|
| Data Loading | 30s | 28s | 1.1x |
| Model Inference (1 split) | 120s | 15s | 8x |
| Grad-CAM Generation | 300s | 60s | 5x |
| **Total (5 splits)** | ~2000s | ~400s | **5x** |

Actual speedup depends on GPU model and dataset size.

## GPU Memory Management

The CUDA notebook automatically:

1. **Clears GPU cache** between splits
2. **Monitors memory allocation** throughout execution
3. **Uses optimal batch sizes** (32 for GPU vs 16 for CPU)
4. **Implements error handling** with automatic cleanup on failure

### Manual GPU Memory Management

If needed, manually clear GPU memory in a notebook cell:

```python
import torch
torch.cuda.empty_cache()
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Advanced Configuration

### cuDNN Benchmark Mode

For faster inference (might use more memory):

```python
torch.backends.cudnn.benchmark = True
```

### Deterministic Behavior

For reproducible results (slower):

```python
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

### Custom CUDA Device

To use a specific GPU:

```python
device = torch.device("cuda:1")  # Use second GPU
```

## Installation Commands Summary

**Complete CUDA setup in one go:**

```powershell
py -m pip install --upgrade pip setuptools wheel
py -m pip install nvidia-cudnn-cu12 --extra-index-url https://download.pytorch.org/whl/torch_stable.html
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## Additional Resources

- **NVIDIA CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **PyTorch CUDA Documentation:** https://pytorch.org/docs/stable/notes/cuda.html
- **cuDNN Documentation:** https://docs.nvidia.com/deeplearning/cudnn/

## Notes

- CUDA 12.1 is the current stable version used by PyTorch 2.9.1
- This setup works on Windows 10/11 with Python 3.14+
- GPU acceleration is optional; CPU processing works fine for smaller datasets
- Both notebooks produce identical results; only performance differs
