# SUSD: Structured Unsupervised Skill Discovery through State Factorization

🎉 **We are pleased to announce that our paper has been accepted to the ICLR 2026 Main Conference.**

This repository contains the official implementation of **SUSD: Structured Unsupervised Skill Discovery through State Factorization**.  
📄 [Paper](https://openreview.net/forum?id=INr5TSooxR)


The implementation is based on
[METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://github.com/seohongpark/METRA).


## Requirements
- Python 3.8

## Installation

```
conda create --name dsd python=3.8
conda activate dsd
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
# In case of C++ error:
# conda install -c conda-forge swig
cd envs/Pettingzoo-skill
pip install -e .
```

Additional Installation
```
pip install slot_attention patchelf
```

Mujoco
```
tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```