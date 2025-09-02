#!/bin/bash
# Setup script for Player Selection Network project

echo "Setting up Player Selection Network environment..."

# Create conda environment
echo "Creating conda environment 'player_selection'..."
conda create -n player_selection python=3.10 -y

# Activate environment
echo "Activating environment..."
conda activate player_selection

# Install core packages via conda
echo "Installing core packages via conda..."
conda install -c conda-forge numpy scipy matplotlib jupyter ipykernel -y

# Install JAX with CUDA support
echo "Installing JAX with CUDA support..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other packages via pip
echo "Installing additional packages via pip..."
pip install flax optax torch torchvision torchaudio tensorboard tqdm pyyaml requests lqrax

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate player_selection"
