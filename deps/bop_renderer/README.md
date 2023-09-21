# BOP renderer

A simple C++ renderer with Python bindings utilized in the BOP toolkit.

The renderer is based on OSMesa, an off-screen rendering library, which makes it suitable for rendering on servers.

### Dependences

The BOP renderer depends on [OSMesa](https://www.mesa3d.org/osmesa.html) which requires [LLVM](https://llvm.org/). You can install OSMesa and LLVM using the provided script *osmesa-install/osmesa-install.sh*, which is a modified version of script [osmesa-install](https://github.com/devernay/osmesa-install) (the changes are documented in *osmesa-install/README.md*).

The installation locations can be set by *osmesaprefix* and *llvmprefix* in *osmesa-install.sh*. If you do not want to install LLVM, set *buildllvm* to 0 in osmesa-install.sh.

To install OSMesa and LLVM, go to folder *bop_renderer* and run:

```
mkdir osmesa-install/build
cd osmesa-install/build
../osmesa-install.sh
```

On Debian/Ubuntu systems, this is also available through `sudo apt install libosmesa6-dev`.

Moreover, the BOP renderer depends on the following header-only libraries, which are provided in folder *3rd* (no installation is required for these libraries): [glm](https://glm.g-truc.net/0.9.9/index.html), [lodepng](https://lodev.org/lodepng/), [pybind11](https://github.com/pybind/pybind11), [RPly](http://w3.impa.br/~diego/software/rply/).

### Compilation

Compile by:
```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Note: The BOP renderer was tested on Linux only.

### Samples

- *samples/renderer_minimal.py* - A minimal example on how to use the Python bindings of the BOP renderer.
- *samples/renderer_test.py* - Comparison of the BOP renderer and the Python renderer from the BOP toolkit.
