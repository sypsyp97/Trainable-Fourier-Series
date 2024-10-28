## Installation Guide for HPC

### Quick Installation

1. **Download and Install Anaconda**:
   Start by downloading the Anaconda distribution for your operating system from
   the [official Anaconda website](https://www.anaconda.com/products/distribution#download-section).

2. **Create a Virtual Environment**:
   Once you've installed Anaconda, create a virtual environment for your project. This will help you manage dependencies
   and avoid conflicts.

   ```bash
   conda create --name pyronn python=3.11
   ```

3. **Activate the Virtual Environment**:

   ```bash
   conda activate pyronn
   ```

4. **Install the CUDA Toolkit**:
   To make use of NVIDIA's CUDA capabilities, install the CUDA toolkit specific to version `11.8.0`:

   ```bash
   conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
   ```

5. **Install PyTorch and Associated Libraries**:
   You can install PyTorch, torchvision, and torchaudio for the specified CUDA version:

   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   ```

6. **Install Pyro-NN**:
   Finally, install the `pyronn` library using the provided wheel file, you can get the wheel file from Yipeng or Linda.

   ```bash
   pip install pyronn-0.3.1-cp311-cp311-linux_x86_64.whl
   ```

### Build by yourself on HPC


1. **Create a Virtual Environment**:
   Once you've installed Anaconda, create a virtual environment for your project. This will help you manage dependencies
   and avoid conflicts.

   ```bash
   conda create --name pyronn python=3.11
   ```

2. **Activate the Virtual Environment**:

   ```bash
   conda activate pyronn
   ```

3. **Install the CUDA Toolkit**:
   To make use of NVIDIA's CUDA capabilities, install the CUDA toolkit specific to version `11.8.0`:

   ```bash
   conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
   ```

4. **Install PyTorch and Associated Libraries**:
   You can install PyTorch, torchvision, and torchaudio for the specified CUDA version:

   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   ```
5. **Build Package**: Install the build package.
   ```bash
    pip install build
    ```
6. **C++ Compiler**: Install the C++ compiler.

   ```bash
   conda install gxx_linux-64
   ```
7. **Build the Pyro-NN package**: Submit the build job.

   ```bash
   sbatch build.sh
   ```
   The content of `build.sh` is:
   ```bash
   #!/bin/bash -l

   ### current working directory
   #SBATCH --chdir=/mnt/home/your_user_name/pyro-nn-torch/
   
   #SBATCH --partition=gpu_A5000
   #SBATCH --job-name=build
   #SBATCH --time=24:00:00
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=1
   #SBATCH --output=/mnt/home/your_user_name/pyro-nn-torch/slurm/%j_%x_%Y%m%d.out
   #SBATCH --error=/mnt/home/your_user_name/pyro-nn-torch/slurm/%j_%x_%Y%m%d_error.out
   
   # Activate Conda environment
   eval "$(conda shell.bash hook)"
   conda activate pyronn
   
   # run with the specific Python interpreter from the pyronn environment
   srun /mnt/home/your_user_name/.conda/envs/pyronn/bin/python -m build
   ```

8. **Wheel File**: Post-build, locate the wheel file in the `dist` directory.

> **Tip**: Adjust `pyproject.toml` if you need a different torch and CUDA version.

