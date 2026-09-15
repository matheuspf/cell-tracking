#!/usr/bin/env bash
# Source this from bash or zsh; all mutable runtime state uses the permanent disk.
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/workspace/tmp
export XDG_CACHE_HOME=/workspace/cache
export XDG_CONFIG_HOME=/workspace/home/root/.config
export XDG_DATA_HOME=/workspace/home/root/.local/share
export XDG_STATE_HOME=/workspace/home/root/.local/state
export JUPYTER_CONFIG_DIR=/workspace/home/root/.jupyter
export JUPYTER_DATA_DIR=/workspace/home/root/.local/share/jupyter
export JUPYTER_RUNTIME_DIR=/workspace/tmp/jupyter
export IPYTHONDIR=/workspace/home/root/.ipython
export CONDA_ENVS_PATH=/workspace/conda/envs:/workspace/kaggle/envs
export CONDA_PKGS_DIRS=/workspace/cache/conda/pkgs
export PIP_CACHE_DIR=/workspace/cache/pip
export UV_CACHE_DIR=/workspace/cache/uv
export npm_config_cache=/workspace/cache/npm
export npm_config_prefix=/workspace/toolchain/npm-global
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export CUDA_CACHE_PATH=/workspace/cache/cuda
export TRITON_CACHE_DIR=/workspace/cache/triton
export NUMBA_CACHE_DIR=/workspace/cache/numba
export MPLCONFIGDIR=/workspace/cache/matplotlib
export PLAYWRIGHT_BROWSERS_PATH=/workspace/cache/ms-playwright
export KAGGLE_CONFIG_DIR=/workspace/home/root/.kaggle
export EDITOR=vim
export PATH=/workspace/bin:/workspace/toolchain/cua_node/bin:/workspace/miniconda3/condabin:$PATH

if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
    . /workspace/miniconda3/etc/profile.d/conda.sh
    if { [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" = base ]; } && [ -x /workspace/conda/envs/cell-tracking/bin/python ]; then
        conda activate cell-tracking
    fi
fi
