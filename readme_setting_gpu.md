## Check NVCC Compatibility
Check your nvcc version (if any) by running `nvcc -V`.

## Install PyTorch versions
First install pytorch binaries with specific version of cuda. We will use `pytorch 1.7.1 + CUDA 10.1` as our example. 
```
torch 1.8.X+cu102/cu111             ==> CUDA 10.2 / 11.1
torch 1.7.X+cu92/cu101/cu102/cu110  ==> CUDA 9.2 / 10.1 / 10.2 / 11.0
torch 1.6.X+cu92/cu101/cu102        ==> CUDA 9.2 / 10.1 / 10.2

- https://pytorch.org/get-started/previous-versions/
```

## Find out usable GPU's
CUDA uses different numbering gpu's than those of `nvidia-smi`.
To find out the correct gpu numbers to set in **CUDA_VISIBLE_DEVICES**

In the file `scripts\8_imv_lstm_attention.ipynb` or `scripts\8_imv_lstm_attention.py`  
os.environ["CUDA_VISIBLE_DEVICES"] = `"0"` (gpu number)

 Otherwise, using CPU set to `"-1"` (If the value is changed, a restart is required.)


