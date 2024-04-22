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
from pyronn.ct_reconstruction.helpers.filters import filters, weights
from pyronn.ct_reconstruction.helpers.phantoms.shepp_logan import shepp_logan_enhanced
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
    circular_trajectory_2d,
)
from pyronn.ct_reconstruction.layers.backprojection_2d import FanBackProjection2D
from pyronn.ct_reconstruction.layers.projection_2d import FanProjection2D


def example_fan_2d_shortscan():
    # Volume Parameters:
    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = [500]
    detector_spacing = [1]

    # Trajectory Parameters:
    number_of_projections = 250

    sdd = 1200
    sid = 750

    angular_range = np.pi + 2 * np.arctan(
        ((detector_shape[0] - 1) / 2.0 * detector_spacing[0]) / sdd
    )

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
        source_isocenter_distance=sid,
        source_detector_distance=sdd,
    )

    # Create Phantom
    phantom = shepp_logan_enhanced(volume_shape)
    # Add required batch dimension
    phantom = torch.tensor(
        np.expand_dims(phantom, axis=0).copy(), dtype=torch.float32
    ).cuda()
    # Build up Reconstruction Pipeline

    # Create Sinogram of Phantom
    sinogram = FanProjection2D().forward(phantom, **geometry)

    # Redundancy Weighting: Create Weight Image and point-wise multiply
    redundancy_weights = torch.tensor(
        weights.parker_weights_2d(geometry), dtype=torch.float32
    ).cuda()
    sinogram_redun_weighted = sinogram * redundancy_weights

    # Filtering: Create 2D Filter and pointwise multiply
    reco_filter = torch.tensor(
        filters.ram_lak_2D(
            geometry.detector_shape,
            geometry.detector_spacing,
            geometry.number_of_projections,
        )
    ).cuda()
    x = torch.fft.fft(sinogram_redun_weighted, dim=-1, norm="ortho")
    x = torch.multiply(x, reco_filter)
    x = torch.fft.ifft(x, dim=-1, norm="ortho").real
    # Final Backprojection
    reco = FanBackProjection2D().forward(x.contiguous(), **geometry)
    reco = reco.cpu().numpy()
    plt.figure()
    plt.imshow(np.squeeze(reco), cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    example_fan_2d_shortscan()
