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
from pyronn.ct_reconstruction.helpers.filters.filters import shepp_logan_3D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
    circular_trajectory_3d,
)
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D


def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 256
    volume_shape = (volume_size, volume_size, volume_size)
    volume_spacing = (1, 1, 1)

    # Detector Parameters:
    detector_shape = (400, 600)
    detector_spacing = (1, 1)

    # Trajectory Parameters:
    number_of_projections = 180
    angular_range = np.pi

    sdd = 1200
    sid = 750

    # create Geometry class
    geometry = Geometry()
    geometry.init_from_parameters(
        volume_shape=volume_shape,
        volume_spacing=volume_spacing,
        detector_shape=detector_shape,
        detector_spacing=detector_spacing,
        number_of_projections=number_of_projections,
        angular_range=angular_range,
        trajectory=circular_trajectory_3d,
        source_isocenter_distance=sid,
        source_detector_distance=sdd,
    )

    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    # Add required batch dimension
    phantom = torch.tensor(
        np.expand_dims(phantom, axis=0).copy(), dtype=torch.float32
    ).cuda()

    # ------------------ Call Layers ------------------ The following code is the new TF2.0 experimental way to tell
    # Tensorflow only to allocate GPU memory needed rather than allocate every GPU memory available. This is
    # important for the use of the hardware interpolation projector, otherwise there might be no enough memory left
    # to allocate the texture memory on the GPU

    sinogram = ConeProjection3D().forward(phantom, **geometry)

    reco_filter = torch.tensor(
        shepp_logan_3D(
            geometry.detector_shape,
            geometry.detector_spacing,
            geometry.number_of_projections,
        ),
        dtype=torch.float32,
    ).cuda()
    x = torch.fft.fft(sinogram, dim=-1, norm="ortho")
    x = torch.multiply(x, reco_filter)
    x = torch.fft.ifft(x, dim=-1, norm="ortho").real

    reco = ConeBackProjection3D().forward(x.contiguous(), **geometry)
    reco = reco.cpu().numpy()
    plt.figure()
    plt.imshow(np.squeeze(reco)[volume_shape[0] // 2], cmap="gray")
    plt.axis("off")
    plt.savefig("cone_3d.png", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    example_cone_3d()
