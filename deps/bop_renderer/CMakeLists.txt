cmake_minimum_required(VERSION 3.16)
project(bop_renderer LANGUAGES C CXX)

set(CMAKE_MODULE_PATH CMake $CMAKE_MODULE_PATH)

find_package(Python COMPONENTS Interpreter Development)
find_package(OSMesa REQUIRED)

add_subdirectory(3rd/pybind11)

# Header files.
set(HEADERFILES
        src/Renderer.h
        src/PythonWrapper.h
        src/AbstractLoader.h
        src/PlyLoader.h
        src/GPUBuffer.h
        src/Pose.h
        src/Geometry.h
        src/glutils/FrameBufferObject.h
        src/glutils/GLSLProgram.h
        src/glutils/gl_core_3_3.h
        src/context/Context.h
        src/Texture.h
        src/Shaders.h
        )

# Source files.
set(SOURCEFILES
        src/Renderer.cpp
        src/AbstractLoader.cpp
        src/PlyLoader.cpp
        src/GPUBuffer.cpp
        src/Pose.cpp
        src/glutils/FrameBufferObject.cpp
        src/glutils/GLSLProgram.cpp
        src/glutils/gl_core_3_3.cpp
        src/context/OSMContext.cpp
        3rd/rply/rply.c
        3rd/lodepng/lodepng.cpp
        )

# Python module.
pybind11_add_module(bop_renderer MODULE
        src/PythonWrapper.cpp ${HEADERFILES} ${SOURCEFILES})

target_link_libraries(bop_renderer PRIVATE OSMesa::OSMesa)
target_include_directories(bop_renderer PRIVATE src 3rd 3rd/glm)
