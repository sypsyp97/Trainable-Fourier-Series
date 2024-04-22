# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.filters import filters
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
    circular_trajectory_2d,
)
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjection2D
from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjection2D


def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = [800]
    detector_spacing = [1]

    # Trajectory Parameters:
    number_of_projections = 360
    angular_range = 2 * np.pi

    # create Geometry class
    geometry = Geometry()
    geometry.init_from_parameters(
        volume_shape=volume_shape,
        volume_spacing=volume_spacing,
        detector_shape=detector_shape,
        detector_spacing=detector_spacing,
        number_of_projections=number_of_projections,
        angular_range=angular_range,
        trajectory=circular_trajectory_2d,
    )

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = torch.tensor(
        np.expand_dims(phantom, axis=0).copy(), dtype=torch.float32
    ).cuda()

    # ------------------ Call Layers ------------------
    sinogram = ParallelProjection2D().forward(phantom, **geometry)

    # sinogram = sinogram + np.random.normal(
    #    loc=np.mean(np.abs(sinogram)), scale=np.std(sinogram), size=sinogram.shape) * 0.02

    reco_filter = torch.tensor(
        filters.shepp_logan_2D(
            geometry.detector_shape,
            geometry.detector_spacing,
            geometry.number_of_projections,
        )
    ).cuda()
    x = torch.fft.fft(sinogram, dim=-1, norm="ortho")

    x = torch.multiply(x, reco_filter)
    x = torch.fft.ifft(x, dim=-1, norm="ortho").real

    reco = ParallelBackProjection2D().forward(x.contiguous(), **geometry)

    reco = reco.cpu().numpy()

    plt.figure()
    plt.imshow(np.squeeze(reco), cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure()
    plt.imshow(np.squeeze(reco) - phantom.cpu().numpy().squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    example_parallel_2d()
