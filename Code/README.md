# DL Evaluation Notebooks

This folder contains all necessary files for running deep learning model evaluation with support for both CPU and GPU/CUDA acceleration.

## Quick Start

### 1. Choose Your Notebook

- **`dl_eval_notebook.ipynb`** - CPU-only version (recommended for development)
- **`dl_eval_notebook_cuda.ipynb`** - GPU/CUDA-optimized version (for production with NVIDIA GPU)

Both notebooks produce identical results; only performance differs.

### 2. Setup CUDA (Optional)

If you have an NVIDIA GPU and want to use GPU acceleration, follow the steps in `SETUP_GUIDE_CUDA.md`.

**Quick CUDA setup:**
```powershell
py -m pip install --upgrade pip setuptools wheel
py -m pip install nvidia-cudnn-cu12 --extra-index-url https://download.pytorch.org/whl/torch_stable.html
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run a Notebook

**From PowerShell in the Training directory:**

```powershell
# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

# Run CPU notebook (default)
jupyter notebook notebooks/dl_eval_notebook.ipynb

# OR run CUDA notebook (if CUDA is installed)
jupyter notebook notebooks/dl_eval_notebook_cuda.ipynb
```

## Files in This Folder

| File | Purpose |
|------|---------|
| `dl_eval_notebook.ipynb` | CPU-only evaluation notebook |
| `dl_eval_notebook_cuda.ipynb` | GPU/CUDA-optimized evaluation notebook |
| `dl_funcs_notebook.py` | Supporting functions module (required by both notebooks) |
| `SETUP_GUIDE_CUDA.md` | Complete CUDA/cuDNN installation guide |
| `README.md` | This file |

## Key Features

✅ **Automatic Device Detection** - Runs on GPU if available, falls back to CPU
✅ **GPU Memory Optimization** - Automatic memory management between splits
✅ **GPU Metrics Display** - Shows GPU info, memory usage, cuDNN status
✅ **Error Handling** - Graceful handling of GPU errors with auto-cleanup
✅ **Batch Size Optimization** - Uses larger batches on GPU (32) vs CPU (16)
✅ **5-Split Cross-Validation** - Automatic split loop with result aggregation
✅ **Grad-CAM Visualization** - GPU-accelerated interpretability analysis

## Notebook Structure

Both notebooks follow the same structure:

1. **Section 1: Import & Setup** - Load libraries and check device
2. **Section 2: Configuration** - Define paths and parameters
3. **Section 3: Data Loading** - Load or generate dataset
4. **Section 4: Evaluation** - Run 5-split evaluation loop
5. **Section 5: Grad-CAM** - Generate interpretability visualizations
6. **Section 6: Results** - Display aggregated statistics

## Performance Notes

Expected speedup with GPU (example: RTX 3090):
- Data Loading: 1.1x
- Model Inference: 8x
- Grad-CAM Generation: 5x
- **Total (5 splits): 5x faster**

Actual speedup depends on your GPU model and dataset size.

## Troubleshooting

### CUDA Not Detected?
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

If this returns `False`:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall cuDNN: See `SETUP_GUIDE_CUDA.md`
3. Reinstall PyTorch with CUDA 12.1 support

### Out of Memory Errors?
- Reduce batch size in Section 2 configuration
- Use CPU notebook instead
- Reduce `num_samples` for Grad-CAM

### Import Errors in Notebook?
Ensure `dl_funcs_notebook.py` is in the same folder as the notebooks.

## Advanced Configuration

### Custom GPU Device
In the configuration section, modify:
```python
device = torch.device("cuda:0")  # Use specific GPU
```

### Enable Deterministic Behavior
```python
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

### Manual GPU Memory Cleanup
```python
import torch
torch.cuda.empty_cache()
```

## Next Steps

1. Review `SETUP_GUIDE_CUDA.md` if you have a GPU
2. Open your chosen notebook in Jupyter
3. Update paths in Section 2 to match your data locations
4. Run cells sequentially

## Support

For issues or questions:
- Check `SETUP_GUIDE_CUDA.md` for CUDA setup help
- Review notebook cell outputs for error messages
- Refer to PyTorch documentation: https://pytorch.org/docs/stable/index.html

## Notes

- Both notebooks can run simultaneously (one on CPU, one on GPU)
- Results are saved to `output_base_dir` (configurable in Section 2)
- GPU version automatically falls back to CPU if CUDA unavailable
- CPU and GPU versions produce identical numerical results
