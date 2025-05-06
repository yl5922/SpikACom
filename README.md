# SpikACom: A Neuromorphic Computing Framework for Green Communication

This repository contains the simulation code for the paper:  
**"SpikACom: A Neuromorphic Computing Framework for Green Communication."**

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

SpikACom is a neuromorphic computing framework designed to enable energy-efficient and adaptive communication by leveraging spiking neural networks (SNNs). This repository provides the simulation code used in our paper to evaluate the performance of SpikACom in three representative communication scenarios, including task-oriented semantic communications, MIMO beamforming, and OFDM channel estimation.

## Dependencies

The code has been tested in the following environment:

- **Python** = 3.10.13
- **SpikingJelly** = 0.0.0.0.14
- **NumPy** = 1.24.3
- **SciPy** = 1.15.1
- **h5py** = 3.5.0

To install the required dependencies, run:

```sh
pip install spikingjelly==0.0.0.0.14 numpy==1.24.3 scipy==1.15.1 h5py==3.5.0
```
Or create a virtual environment:

```sh
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install spikingjelly==0.0.0.0.14 numpy==1.24.3 scipy==1.15.1 h5py==3.5.0
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

- 📄 Paper Link: https://arxiv.org/abs/2502.17168

---

We appreciate your interest in this work and look forward to your feedback! 🚀
