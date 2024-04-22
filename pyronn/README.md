# PYRO-NN Framework

Welcome to PYRO-NN, a Python framework for state-of-the-art reconstruction algorithms seamlessly integrated into
PyTorch. This library serves as a bridge to the layers implemented
in [PYRO-NN-Layers](https://github.com/csyben/PYRO-NN-Layers).

## Introduction

PYRO-NN is designed to bring cutting-edge reconstruction techniques to neural networks. The open-access paper detailing
its capabilities and design can be accessed [here](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13753).

**Dependencies**: PYRO-NN relies on the `pyronn_layers`. These are now available through pip. For those interested in
the source code, it's available at [PYRO-NN-Layers](https://github.com/csyben/PYRO-NN-Layers).

## References

If you find PYRO-NN beneficial for your research or applications, please consider citing our work:

```bibtex
@article{PYRONN2019,
   author = {Syben, Christopher and Michen, Markus and Stimpel, Bernhard and Seitz, Stephan and Ploner, Stefan and Maier, Andreas K.},
   title = {Technical Note: PYRO-NN: Python reconstruction operators in neural networks},
   year = {2019},
   journal = {Medical Physics},
}
```

## Installation Guide

### Quick Installation

1. Download and Install Anaconda:
   Start by downloading the Anaconda distribution for your operating system from
   the [official Anaconda website](https://www.anaconda.com/products/distribution#download-section).


2. Create a Virtual Environment:
   Once you've installed Anaconda, create a virtual environment for your project. This will help you manage dependencies
   and avoid conflicts.

   ```bash
   conda create --name your_env_name python=3.11
   ```

3. Activate the Virtual Environment:

   ```bash
   conda activate your_env_name
   ```

4. Install the CUDA Toolkit:
   To make use of NVIDIA's CUDA capabilities, install the CUDA toolkit specific to version `11.8.0`:

   ```bash
   conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
   ```

5. Install PyTorch and Associated Libraries:
   You can install PyTorch, torchvision, and torchaudio for the specified CUDA version:

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

6. Install Pyronn:
   Finally, install the `pyronn` library using the provided [wheel file](wheel/pyronn-0.3.1-cp311-cp311-win_amd64.whl):

   ```bash
   pip install pyronn-0.3.1-cp311-cp311-win_amd64.whl
   ```

### Manual Installation

For a hands-on installation:

#### Prerequisites:

- **Microsoft Visual C++**: Version 14.0 or higher.
- **Build Package**: Essential for the build process.
- **CUDA**: Ensure you have version 10.2 or later.

#### Steps:


1. **Version Check**: Ensure the torch version in `pyproject.toml` aligns with your environment to avoid DLL errors and
   install C++ Compiler:

   ```bash
   conda install gxx_linux-64
   ```
2. **Package Build**:

   ```bash
   python -m build
   ```

3. **Wheel File**: Post-build, locate the wheel file in the `wheel` directory.

> **Tip**: Adjust `pyproject.toml` if you need a different torch version.

## Basic Example

This guide demonstrates the foundational steps for utilizing PYRO-NN layers, using a simplified example. We'll initiate
with the creation of a sinogram and its corresponding geometry.

```python
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d

# Volume Configuration:
volume_size = 256
volume_shape = [volume_size, volume_size]
volume_spacing = [1, 1]

# Detector Configuration:
detector_shape = [800]
detector_spacing = [1]

# Trajectory Configuration:
number_of_projections = 360
angular_range = 2 * np.pi

# Initialize Geometry:
geometry = Geometry()
geometry.init_from_parameters(volume_shape=volume_shape, volume_spacing=volume_spacing,
                              detector_shape=detector_shape, detector_spacing=detector_spacing,
                              number_of_projections=number_of_projections, angular_range=angular_range,
                              trajectory=circular_trajectory_2d)
```

To leverage the capabilities of PYRO-NN, instantiate the `Geometry` class. This feeds the system essential geometric
information. If you lack a scanner, simulate one with PYRO-NN.

```python
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjection2D

# Create a Phantom:
phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
phantom = torch.tensor(np.expand_dims(phantom, axis=0).copy(), dtype=torch.float32).cuda()

# Generate Sinogram:
sinogram = ParallelProjection2D().forward(phantom, **geometry)
```

Ensure that the geometry for both projection and reconstruction is consistent. The
function `ParallelProjection2d().forward(phantom, **geometry)` emulates a scanner.

Before implementing the Filtered Back Projection (FBP) algorithm, pre-process the sinogram:

```python
import torch
from pyronn.ct_reconstruction.helpers.filters import filters

# Apply Filter:
reco_filter = torch.tensor(
    filters.shepp_logan_2D(geometry.detector_shape, geometry.detector_spacing, geometry.number_of_projections)).cuda()
x = torch.fft.fft(sinogram, dim=-1, norm="ortho")
x = torch.multiply(x, reco_filter)
x = torch.fft.ifft(x, dim=-1, norm="ortho").real
```

Explore various filters in the `filters` module.

Post-processing is straightforward; retrieve the reconstruction with a single line:

```python
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjection2D

# Obtain Reconstruction:
reco = ParallelBackProjection2D().forward(x.contiguous(), **geometry)
reco = reco.cpu().numpy()
```

Note: PYRO-NN bifurcates projection and reconstruction into distinct classes. Always remember to detach the GPU
post-processing.

## Geometry Overview

The `Geometry` class offers initialization from parameters, JSON files, or EZRT files. The exact geometry type hinges on
the provided trajectory. Parameters are cataloged within the `parameter_dict` dictionary.

### Properties of Geometry:

| Property                  | Description                                                                         |
|---------------------------|-------------------------------------------------------------------------------------|
| volume_shape              | [volume_Z, volume_X, Volume_Y]                                                      |
| volume_spacing            | Axis-specific spacing                                                               |
| volume_origin             | Volume origin coordinates                                                           |
| detector_shape            | [detector_height, detector_width]                                                   |
| detector_spacing          | Detector spacing                                                                    |
| detector_origin           | Detector center coordinates                                                         |
| number_of_projections     | Total projection count                                                              |
| angular_range             | Either a 2-element list or a singular float. Single values yield a range [0, value] |
| sinogram_shape            | Automatically derived sinogram shape                                                |
| source_detector_distance  | Non-pixel specific distance between source and detector                             |
| source_isocenter_distance | Non-pixel specific distance between source and iso-center                           |
| trajectory                | Resultant trajectory computation                                                    |
| projection_multiplier     | Auto-calculated multiplier                                                          |
| step_size                 | Sampling step size (default: 0.2)                                                   |

### Geometry Class Methods:

| Method             | Description                           |
|--------------------|---------------------------------------|
| fan_angle          | Retrieve trajectory angle values      |
| cone_angle         | Retrieve trajectory angle values      |
| set_detector_shift | Adjust the origin if required         |
| set_volume_slice   | Currently non-operational             |
| set_angle_range    | Modify the projection angle           |
| swap_axis          | Set system rotation direction         |
| slice_the_geometry | Segment geometry into smaller subsets |

---

## Limitations

The image reconstructed using backprojection will exhibit a difference of approximately 1000 magnitudes compared to the
original image.

---

## Changelog

For detailed updates and modifications, refer to the [CHANGELOG.md](CHANGELOG.md).

---

## License

This project abides by the [Apache-2.0 License](LICENSE).


