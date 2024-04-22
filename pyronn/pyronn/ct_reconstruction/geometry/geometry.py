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
import copy
import warnings
from importlib.util import find_spec
from typing import Callable, Tuple, Any

import numpy as np
import torch

from pyronn.ct_reconstruction.helpers.trajectories.arbitrary_trajectory import (
    arbitrary_projection_matrix,
)
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
    circular_trajectory_3d,
)

if find_spec("PythonTools"):
    pass

"""
try:
    from PythonTools import ezrt_header
except ImportError as e:
    pass
"""


class Geometry:
    """
    The Base Class for the different Geometries. Provides commonly used members.
    """

    def __init__(self):
        self.gpu_device = True
        self.traj_func = None  # Placeholder
        self.np_dtype = np.float32
        self.parameter_dict = {
            "volume_shape": None,
            "volume_spacing": None,
            "volume_origin": None,
            "detector_shape": None,
            "detector_spacing": None,
            "detector_origin": None,
            "number_of_projections": None,
            "angular_range": None,
            "trajectory": None,
            "source_detector_distance": None,
            "source_isocenter_distance": None,
            "projection_multiplier": None,
            "step_size": None,
            "swap_detector_axis": False,
            "sinogram_shape": None,
        }

    def init_from_parameters(
        self,
        volume_shape: Tuple[int, ...],
        volume_spacing: Tuple[float, ...],
        detector_shape: Tuple[int, ...],
        detector_spacing: Tuple[float, ...],
        number_of_projections: int,
        angular_range: Tuple[float, ...],
        trajectory: Callable,
        source_detector_distance: float = 0.0,
        source_isocenter_distance: float = 0.0,
        swap_detector_axis: bool = False,
    ) -> None:
        # Helper function to compute origin
        def compute_origin(shape, spacing):
            return -(shape - 1) / 2.0 * spacing

        # Assigning simple values
        self.parameter_dict["swap_detector_axis"] = swap_detector_axis
        self.parameter_dict["number_of_projections"] = number_of_projections
        self.parameter_dict["source_detector_distance"] = source_detector_distance
        self.parameter_dict["source_isocenter_distance"] = source_isocenter_distance
        self.parameter_dict["angular_range"] = (
            angular_range if isinstance(angular_range, list) else [0, angular_range]
        )

        # Volume Parameters:
        self.parameter_dict["volume_shape"] = np_array = np.array(volume_shape)
        self.parameter_dict["volume_spacing"] = np_spacing = np.array(
            volume_spacing, dtype=self.np_dtype
        )
        self.parameter_dict["volume_origin"] = compute_origin(np_array, np_spacing)

        # Detector Parameters:
        self.parameter_dict["detector_shape"] = np_array = np.array(detector_shape)
        self.parameter_dict["detector_spacing"] = np_spacing = np.array(
            detector_spacing, dtype=self.np_dtype
        )
        self.parameter_dict["detector_origin"] = compute_origin(np_array, np_spacing)

        # Trajectory Parameters:
        self.parameter_dict["sinogram_shape"] = np.array(
            [number_of_projections, *detector_shape]
        )
        self.parameter_dict["trajectory"] = trajectory(**self.parameter_dict)
        self.traj_func = trajectory

        # Containing the constant part of the distance weight and discretization invariant
        self.parameter_dict["projection_multiplier"] = (
            source_isocenter_distance
            * source_detector_distance
            * detector_spacing[-1]
            * np.pi
            / number_of_projections
        )
        self.parameter_dict["step_size"] = 0.2

    def to_json(self, path: str) -> None:
        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        try:
            with open(path, "w+") as outfile:
                json.dump(self.parameter_dict, outfile, cls=NumpyEncoder)
        except FileNotFoundError:
            print("Error: File not found when trying to save geometry.")
        except FileExistsError:
            print("Error: File already exists when trying to save geometry.")

    @staticmethod
    def from_json(path: str):
        import json

        loaded_geom = Geometry()
        try:
            with open(path, "r") as infile:
                loaded_geom.parameter_dict = json.load(infile)
        except FileNotFoundError:
            print("Error: File not found when trying to load geometry.")

        for key, value in loaded_geom.parameter_dict.items():
            if isinstance(value, list):
                loaded_geom.parameter_dict[key] = np.asarray(value)

        return loaded_geom

    # Thanks to Linda-Sophie Schneider
    def init_from_EZRT_header(
        self,
        projection_headers: Tuple[str, ...],
        reco_header=None,
        volume_spacing=None,
        volume_shape=None,
        detector_shape=None,
        detector_spacing=None,
        swap_detector_axis: bool = True,
        **kwargs,
    ) -> None:
        self.headers = projection_headers
        header = self.headers[0]
        traj_type = (
            "circ"
            if np.array_equal(np.array(header.agv_source_position), np.array([0, 0, 0]))
            else "free"
        )
        self.traj_func = arbitrary_projection_matrix
        print(traj_type)
        if traj_type == "circ":
            angular_range = 2 * np.pi  # full circle
            self.parameter_dict["angular_range"] = [0, angular_range]

        # if reco_header.num_voxel_z != 0 and reco_header.num_voxel_y != 0 and reco_header.num_voxel_x != 0:
        #     self.parameter_dict['volume_shape'] = np.asarray(
        #         [reco_header.num_voxel_z, reco_header.num_voxel_y, reco_header.num_voxel_x])
        # else:
        #     warnings.warn("Warning: No valid volume shape for Reconstruction could be defined in the geometry")

        # if reco_header.voxel_size_z_in_um != 0 and reco_header.voxel_size_x_in_um != 0: self.parameter_dict[
        # 'volume_spacing'] = np.asarray( [reco_header.voxel_size_z_in_um, reco_header.voxel_size_x_in_um,
        # reco_header.voxel_size_x_in_um]) / 1000.0 elif reco_header.voxel_size_in_um != 0: self.parameter_dict[
        # 'volume_spacing'] = np.full(shape=3, fill_value=header.voxel_size_in_um / 1000.0, dtype=self.np_dtype)
        # else: warnings.warn("Warning: No valid volume spacing for Reconstruction could be defined in the geometry")
        scaling_factor = (
            header._focus_detector_distance_in_um / header._focus_object_distance_in_um
        )
        if reco_header == None:
            if volume_spacing == None:
                # Berechnung der Größen für die Rekonstruktion anhand der Header Daten
                pixel = max(
                    [header.number_horizontal_pixels, header.number_vertical_pixels]
                )
                voxelcount = np.ceil(pixel * scaling_factor * (2 / 3))
                # Test if voxelcount is dividable by 8
                if voxelcount % 8 != 0:
                    voxelcount = np.ceil(voxelcount / 8) * 8
                self.parameter_dict["volume_shape"] = np.asarray(
                    [voxelcount, voxelcount, voxelcount]
                )
                self.parameter_dict["volume_spacing"] = (
                    np.asarray(
                        [
                            header.detector_width_in_um / pixel,
                            header.detector_width_in_um / pixel,
                            header.detector_width_in_um / pixel,
                        ]
                    )
                    / 1000.0
                )
            else:
                self.parameter_dict["volume_shape"] = np.asarray(volume_shape)
                self.parameter_dict["volume_spacing"] = np.asarray(
                    [volume_spacing, volume_spacing, volume_spacing]
                )
        elif (
            reco_header.num_voxel_z != 0
            and reco_header.num_voxel_y != 0
            and reco_header.num_voxel_x != 0
        ):
            self.parameter_dict["volume_shape"] = np.asarray(
                [
                    reco_header.num_voxel_z,
                    reco_header.num_voxel_y,
                    reco_header.num_voxel_x,
                ]
            )
            if (
                reco_header.voxel_size_z_in_um != 0
                and reco_header.voxel_size_x_in_um != 0
            ):
                self.parameter_dict["volume_spacing"] = np.asarray(
                    [
                        reco_header.voxel_size_z_in_um,
                        reco_header.voxel_size_x_in_um,
                        reco_header.voxel_size_x_in_um,
                    ]
                ) / (1000.0 / (scaling_factor * 1.11))
            elif reco_header.voxel_size_in_um != 0:
                self.parameter_dict["volume_spacing"] = np.full(
                    shape=3,
                    fill_value=header.voxel_size_in_um / 1000.0,
                    dtype=self.np_dtype,
                )
        else:
            warnings.warn(
                "Warning: No valid volume shape and/or volume spacing for Reconstruction could be defined in the geometry"
            )

        if (
            self.parameter_dict["volume_shape"] is None
            or self.parameter_dict["volume_spacing"] is None
        ):
            warnings.warn(
                "Warning: No valid volume origin for Reconstruction could be computed"
            )
        else:
            self.parameter_dict["volume_origin"] = (
                -(self.parameter_dict["volume_shape"] - 1)
                / 2.0
                * self.parameter_dict["volume_spacing"]
            )

        # Detector Parameters:
        if detector_shape == None:
            self.parameter_dict["detector_shape"] = np.array(
                [header.number_vertical_pixels, header.number_horizontal_pixels]
            )
            self.parameter_dict["detector_spacing"] = np.full(
                shape=2,
                fill_value=(header.detector_height_in_um / 1000.0)
                / header.number_vertical_pixels,
                dtype=self.np_dtype,
            )  # np.array(header.pixel_width_in_um, dtype=self.np_dtype)
        else:
            self.parameter_dict["detector_shape"] = np.array(detector_shape)
            self.parameter_dict["detector_spacing"] = np.full(
                shape=2, fill_value=detector_spacing, dtype=self.np_dtype
            )  # np.array(header.pixel_width_in_um, dtype=self.np_dtype)

        self.parameter_dict["detector_origin"] = (
            -(self.parameter_dict["detector_shape"] - 1)
            / 2.0
            * self.parameter_dict["detector_spacing"]
        )

        # Trajectory Parameters:
        self.parameter_dict["number_of_projections"] = len(
            projection_headers
        )  # -270#___________________________________________________________________________

        self.parameter_dict["sinogram_shape"] = np.array(
            [
                self.parameter_dict["number_of_projections"],
                *self.parameter_dict["detector_shape"],
            ]
        )
        self.parameter_dict[
            "source_detector_distance"
        ] = header.focus_detector_distance_in_mm
        self.parameter_dict[
            "source_isocenter_distance"
        ] = header.focus_object_distance_in_mm
        self.__generate_trajectory__()

        # Containing the constant part of the distance weight and discretization invarian
        self.parameter_dict["projection_multiplier"] = (
            self.parameter_dict["source_isocenter_distance"]
            * self.parameter_dict["source_detector_distance"]
            * self.parameter_dict["detector_spacing"][-1]
            * np.pi
            / self.parameter_dict["number_of_projections"]
        )

        self.parameter_dict["step_size"] = 1.0
        self.parameter_dict["swap_detector_axis"] = swap_detector_axis

    def cuda(self) -> None:
        self.gpu_device = True

    def cpu(self) -> None:
        self.gpu_device = False

    def keys(self) -> dict[Any, Any]:
        return self.parameter_dict

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            parameter = self.parameter_dict[key]
            tensor_value = (
                torch.Tensor(parameter)
                if hasattr(parameter, "__len__")
                else torch.Tensor([parameter])
            )
        except KeyError:
            print(f"Attribute <{key}> could not be transformed to torch.Tensor")
            return torch.Tensor()

        return tensor_value.cuda() if self.gpu_device else tensor_value.cpu()

    # Thanks to Linda-Sophie Schneider
    def __generate_trajectory__(self) -> None:
        if self.traj_func == circular_trajectory_3d:
            self.parameter_dict["trajectory"] = self.traj_func(**self.parameter_dict)
        else:
            self.parameter_dict["trajectory"] = self.traj_func(
                self.headers,
                voxel_size=self.parameter_dict["volume_spacing"],
                swap_detector_axis=self.parameter_dict["swap_detector_axis"],
            )

    def fan_angle(self) -> float:
        return np.arctan(
            (
                (self.parameter_dict["detector_shape"][-1] - 1)
                / 2.0
                * self.parameter_dict["detector_spacing"][-1]
            )
            / self.parameter_dict["source_detector_distance"]
        )

    def cone_angle(self) -> float:
        return np.arctan(
            (
                (self.parameter_dict["detector_shape"][-2] - 1)
                / 2.0
                * self.parameter_dict["detector_spacing"][-2]
            )
            / self.parameter_dict["source_detector_distance"]
        )

    def set_detector_shift(self, detector_shift: Tuple[float, ...]) -> None:
        """
        Applies a detector shift in px to the geometry.
        This triggers a recomputation of the trajectory. Projection matrices will be overwritten.

        :param detector_shift: Tuple[float, ...] With [y,x] convention in Pixels
        """
        # change the origin according to the shift
        self.parameter_dict["detector_origin"] = self.parameter_dict[
            "detector_origin"
        ] + (detector_shift * self.parameter_dict["detector_spacing"])
        # recompute the trajectory with the applied shift
        # TODO: For now cone-beam circular full scan is fixed as initialization. Need to be reworked such that header defines the traj function
        self.__generate_trajectory__()

    def set_volume_slice(self, slice):
        """
        select one slice reconstruction geometry from the 3d objection, this function will correct reconstructions far from center.

        :param slice: selected slice
        :return: a new geometry for selected slice
        """
        # volume_shift = slice - (self.parameter_dict['volume_shape'][0] - 1) / 2.0
        geo = copy.deepcopy(self)
        geo.parameter_dict["volume_origin"][0] = geo.parameter_dict["volume_origin"][
            0
        ] + (slice * geo.parameter_dict["volume_spacing"][0])
        geo.parameter_dict["volume_shape"][0] = 1
        geo.__generate_trajectory__()
        return geo

    def set_angle_range(self, angle_range):
        """
        change the range of angle(geometry). WARNING: this will change the original geometry

        :param angle_range: list or single value, if a single value is given, angle_range will be [0, value]
        :return: None
        """
        if isinstance(angle_range, list):
            self.parameter_dict["angular_range"] = angle_range
        else:
            self.parameter_dict["angular_range"] = [0, angle_range]
        self.__generate_trajectory__()

    def swap_axis(self, swap_det_axis: bool) -> None:
        """
        Sets the direction of the rotatation of the system.
        This triggers a recomputation of the trajectory. Projection matrices will be overwritten.

        :param counter_clockwise: wether the system rotates counter clockwise (True) or not.
        """
        self.parameter_dict["swap_det_axis"] = swap_det_axis
        self.__generate_trajectory__()

    def slice_the_geometry(self, slices):
        """
        Create several(slices) sub-geometries to overcome memory not enough.

        :param: slices: the amount of sub-geometries
        :return: a list of sub-geometries
        """
        nop = self.number_of_projections // slices
        angular_inc = 2 * np.pi / slices
        sliced_geos = []

        for i in range(slices):
            geo = copy.deepcopy(self)
            start_angle = 2 * i * np.pi / slices
            geo.number_of_projections = self.number_of_projections // slices
            geo.set_angle_range([start_angle, start_angle + angular_inc])
            sliced_geos.append(geo)
        return sliced_geos

    @property
    def volume_shape(self) -> Tuple[int, ...]:
        return tuple(self.parameter_dict["volume_shape"])

    @property
    def volume_spacing(self) -> Tuple[float, ...]:
        return tuple(self.parameter_dict["volume_spacing"])

    @property
    def detector_shape(self) -> Tuple[int, ...]:
        return tuple(self.parameter_dict["detector_shape"])

    @property
    def detector_origin(self) -> Tuple[float, ...]:
        return tuple(self.parameter_dict["detector_origin"])

    @property
    def detector_spacing(self) -> Tuple[float, ...]:
        return tuple(self.parameter_dict["detector_spacing"])

    @property
    def number_of_projections(self) -> int:
        return int(self.parameter_dict["number_of_projections"])

    @number_of_projections.setter
    def number_of_projections(self, value):
        self.parameter_dict["number_of_projections"] = value

    @property
    def angular_range(self) -> list or float:
        return self.parameter_dict["angular_range"]

    @property
    def trajectory(self) -> Tuple[float, ...]:
        return self.parameter_dict["trajectory"]

    @property
    def source_detector_distance(self) -> float:
        return self.parameter_dict["source_detector_distance"]

    @property
    def source_isocenter_distance(self) -> float:
        return self.parameter_dict["source_isocenter_distance"]

    @property
    def projection_multiplier(self) -> float:
        return self.parameter_dict["projection_multiplier"]

    @property
    def step_size(self) -> float:
        return self.parameter_dict["step_size"]

    @property
    def swap_detector_axis(self) -> bool:
        return self.parameter_dict["swap_detector_axis"]

    @property
    def is_gpu(self) -> bool:
        return self.gpu_device
