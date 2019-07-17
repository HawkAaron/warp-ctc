import os
import platform
import sys
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = ['-std=c++11', '-fPIC']
warp_ctc_path = "../build"

if torch.cuda.is_available() or "CUDA_HOME" in os.environ:
    enable_gpu = True
else:
    print("Torch was not built with CUDA support, not building warp-ctc GPU extensions.")
    enable_gpu = False

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

build_extension = CppExtension
if enable_gpu:
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']
    from torch.utils.cpp_extension import CUDAExtension
    build_extension = CUDAExtension

if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc" + lib_ext)):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)
include_dirs = [os.path.realpath('../include')]

setup(
    name="warpctc_softmax",
    version="0.1",
    description="PyTorch wrapper for warp-ctc",
    url="https://github.com/baidu-research/warp-ctc",
    author="Jared Casper, Sean Naren",
    author_email="jared.casper@baidu.com, sean.narenthiran@digitalreasoning.com",
    license="Apache",
    packages=find_packages(),
    ext_modules=[
        build_extension(
            name='warpctc_softmax._warp_ctc',
            language='c++',
            sources=['src/binding.cpp'],
            include_dirs=include_dirs,
            library_dirs=[os.path.realpath(warp_ctc_path)],
            libraries=['warpctc'],
            extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_ctc_path)],
            extra_compile_args=extra_compile_args)
        ],
    cmdclass={'build_ext': BuildExtension}
)

