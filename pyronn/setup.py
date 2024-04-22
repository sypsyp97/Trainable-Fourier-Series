__author__ = "Christopher Syben <christopher.syben@fau.de>"
__copyright__ = "Christopher Syben <christopher.syben@fau.de>"
__license__ = """
PYRO-NN, python framework for convenient use of the ct reconstructions algorithms
Copyright [2019] [Christopher Syben]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setuptools.setup(
    ext_modules=[
        CUDAExtension('pyronn_layers', [
            #Python Bindings
            'pyronn/ct_reconstruction/cores/torch_ops/pyronn_torch_layers.cc',
            #Parallel operators
            'pyronn/ct_reconstruction/cores/torch_ops/par_projector_2D_OPKernel.cc', 
            'pyronn/ct_reconstruction/cores/kernels/par_projector_2D_CudaKernel.cu',
            'pyronn/ct_reconstruction/cores/torch_ops/par_backprojector_2D_OPKernel.cc', 
            'pyronn/ct_reconstruction/cores/kernels/par_backprojector_2D_CudaKernel.cu',
            #Fan operators
            'pyronn/ct_reconstruction/cores/torch_ops/fan_projector_2D_OPKernel.cc', 
            'pyronn/ct_reconstruction/cores/kernels/fan_projector_2D_CudaKernel.cu',
            'pyronn/ct_reconstruction/cores/torch_ops/fan_backprojector_2D_OPKernel.cc', 
            'pyronn/ct_reconstruction/cores/kernels/fan_backprojector_2D_CudaKernel.cu',
            # #Cone operators
            'pyronn/ct_reconstruction/cores/torch_ops/cone_projector_3D_OPKernel.cc', 
            'pyronn/ct_reconstruction/cores/kernels/cone_projector_3D_CudaKernel.cu',
            'pyronn/ct_reconstruction/cores/kernels/cone_projector_3D_CudaKernel_hardware_interp.cu',
            'pyronn/ct_reconstruction/cores/torch_ops/cone_backprojector_3D_OPKernel.cc', 
            'pyronn/ct_reconstruction/cores/kernels/cone_backprojector_3D_CudaKernel.cu',
            'pyronn/ct_reconstruction/cores/kernels/cone_backprojector_3D_CudaKernel_hardware_interp.cu',
        ],
        #extra_compile_args=['-g']
        ),
    ],
    cmdclass={
    'build_ext': BuildExtension
})
