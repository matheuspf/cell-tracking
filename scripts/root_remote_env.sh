#!/usr/bin/env bash
# Source from bash/zsh on the Vast host; keep the original /kaggle paths.
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export EDITOR=vim
export TMPDIR=/root/tmp
export CONDA_ENVS_PATH=/root/.conda/envs:/kaggle/envs
export PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
export KAGGLE_CONFIG_DIR=/root/.kaggle
export PATH=/root/bin:/root/toolchain/cua_node/bin:/root/miniconda3/condabin:$PATH

# This image injects toolkit libraries that break the mirrored PyTorch runtime.
# Keep user-supplied library paths, but let the wheel resolve its CUDA libraries.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    remote_cuda_paths=":$LD_LIBRARY_PATH:"
    remote_cuda_paths=${remote_cuda_paths//:\/usr\/local\/cuda\/lib64:/:}
    remote_cuda_paths=${remote_cuda_paths#:}
    remote_cuda_paths=${remote_cuda_paths%:}
    if [ -n "$remote_cuda_paths" ]; then
        export LD_LIBRARY_PATH=$remote_cuda_paths
    else
        unset LD_LIBRARY_PATH
    fi
    unset remote_cuda_paths
fi

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # Vast's .bashrc uses PS1 to identify interactive SSH sessions.
    # Conda must not introduce a prompt into an scp/rsync command shell.
    case $- in *i*) ;; *) export CONDA_CHANGEPS1=false ;; esac
    . /root/miniconda3/etc/profile.d/conda.sh
    if { [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" = base ]; } && [ -x /root/.conda/envs/cell-tracking/bin/python ]; then
        conda activate cell-tracking
    fi
fi
