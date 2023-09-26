import os
from os import path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

here = path.abspath(path.dirname(__file__))

# Use correct conda compiler used to build pytorch
if "GXX" in os.environ:
    os.environ["CXX"] = os.environ.get("GXX", "")

setup(
    name="cosypose",
    version="1.0.0",
    description="CosyPose",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="cosypose_cext",
            sources=["cosypose/csrc/cosypose_cext.cpp"],
            extra_compile_args=["-O3"],
            verbose=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
