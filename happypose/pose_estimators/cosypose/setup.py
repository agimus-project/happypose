from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("cosypose_cext", ["cosypose/csrc/cosypose_cext.cpp"]),
]

setup(
    name="cosypose",
    version="1.0.0",
    description="CosyPose",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
