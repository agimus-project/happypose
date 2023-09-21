#include <iostream>

#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "PythonWrapper.h"

// Ref:
// http://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html?highlight=numpy
// https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11

PythonWrapper::PythonWrapper() {
}

PythonWrapper::~PythonWrapper() {
}

bool PythonWrapper::init(int width, int height) {
  return renderer.init(width, height);
}

void PythonWrapper::setLight(const std::array<float, 3> lightCamPos,
                             const std::array<float, 3> lightColor,
                             float lightAmbientWeight, float lightDiffuseWeight,
                             float lightSpecularWeight,
                             float lightSpecularShininess) {
  glm::vec3 _lightCamPos(lightCamPos[0], lightCamPos[1], lightCamPos[2]);
  glm::vec3 _lightColor(lightColor[0], lightColor[1], lightColor[2]);
  renderer.setLight(_lightCamPos, _lightColor, lightAmbientWeight,
                    lightDiffuseWeight, lightSpecularWeight,
                    lightSpecularShininess);
}

bool PythonWrapper::addObject(unsigned int objId, const std::string &filename) {
  return renderer.addObject(objId, filename);
}

void PythonWrapper::removeObject(unsigned int handle) {
  renderer.removeObject(handle);
}

void PythonWrapper::renderObject(unsigned int handle,
                                 const std::array<float, 9> R,
                                 const std::array<float, 3> t,
                                 float fx, float fy, float cx, float cy,
                                 float skew, float xZero, float yZero,
                                 bool useUniformColor,
                                 float uniformColorR,
                                 float uniformColorG,
                                 float uniformColorB) {
  glm::mat3 _R;
  for (unsigned int x = 0; x < 3; x++) {
    for (unsigned int y = 0; y < 3; y++) {
      _R[x][y] = R[y * 3 + x];
    }
  }

  glm::vec3 _t;
  for (unsigned int x = 0; x < 3; x++) {
    _t[x] = t[x];
  }

  glm::vec3 uniformColor(uniformColorR, uniformColorG, uniformColorB);

  Pose pose(_R, _t);
  renderer.renderObject(handle, pose, fx, fy, cx, cy, skew, xZero, yZero,
                        useUniformColor, uniformColor);
}

py::array PythonWrapper::getDepthImage(unsigned int handle) {

  float *buffData = renderer.getBuffer(handle, OA_DEPTH);

  int height = renderer.getHeight();
  int width = renderer.getWidth();
  int size = height * width;
  auto *arrData = new uint16_t[size];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = x + width * y;
      if (y < 0 || x < 0 || y >= height || x >= width) {
        arrData[i] = 0;
      } else {
        arrData[i] = static_cast<uint16_t>(round(buffData[i]));
      }
    }
  }

  // Create a Python object that will free the allocated memory when destroyed
  // Ref: https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  py::capsule freeWhenDone(arrData, [](void *f) {
    auto *data = reinterpret_cast<uint16_t *>(f);
    delete[] data;
  });

  size_t typeSize = sizeof(uint16_t);
  auto arr = py::array_t<uint16_t>(
      {height, width}, // Shape
      {width * typeSize, typeSize}, // C-style contiguous strides
      arrData, // Data pointer
      freeWhenDone); // Numpy array references this parent

  delete[] buffData;

  return arr;
}

py::array PythonWrapper::getColorImage(unsigned int handle) {

  float *buffData = renderer.getBuffer(handle, OA_COLORS);
//  float *buffData = renderer.getBuffer(handle, OA_TEXTURED);

  int height = renderer.getHeight();
  int width = renderer.getWidth();
  int size = height * width * 3;
  auto *arrData = new uint8_t[size];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = (x + width * y) * 3;
      if (y < 0 || x < 0 || y >= height || x >= width) {
        arrData[i] = 0;
        arrData[i + 1] = 0;
        arrData[i + 2] = 0;
      } else {
        arrData[i] = static_cast<uint8_t>(buffData[i] * 255.0f);
        arrData[i + 1] = static_cast<uint8_t>(buffData[i + 1] * 255.0f);
        arrData[i + 2] = static_cast<uint8_t>(buffData[i + 2] * 255.0f);
      }
    }
  }

  // Create a Python object that will free the allocated memory when destroyed
  // Ref: https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  py::capsule freeWhenDone(arrData, [](void *f) {
    auto *data = reinterpret_cast<uint8_t *>(f);
    delete[] data;
  });

  size_t typeSize = sizeof(uint8_t);
  auto arr = py::array_t<uint8_t>(
      {height, width, 3}, // Shape
      {width * 3 * typeSize, 3 * typeSize, typeSize}, // Strides
      arrData, // Data pointer
      freeWhenDone); // Numpy array references this parent

  delete[] buffData;

  return arr;
}

py::array PythonWrapper::getLocalPosImage(unsigned int handle) {

  float *buffData = renderer.getBuffer(handle, OA_LOCALPOS);

  int height = renderer.getHeight();
  int width = renderer.getWidth();
  int size = height * width * 3;
  auto *arrData = new float_t[size];

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = (x + width * y) * 3;
      if (y < 0 || x < 0 || y >= height || x >= width) {
        arrData[i] = 0;
        arrData[i + 1] = 0;
        arrData[i + 2] = 0;
      } else {
        arrData[i] = static_cast<float_t>(buffData[i]);
        arrData[i + 1] = static_cast<float_t>(buffData[i + 1]);
        arrData[i + 2] = static_cast<float_t>(buffData[i + 2]);
      }
    }
  }

  // Create a Python object that will free the allocated memory when destroyed
  // Ref: https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  py::capsule freeWhenDone(arrData, [](void *f) {
    auto *data = reinterpret_cast<float_t *>(f);
    delete[] data;
  });

  size_t typeSize = sizeof(float_t);
  auto arr = py::array_t<float_t>(
      {height, width, 3}, // Shape
      {width * 3 * typeSize, 3 * typeSize, typeSize}, // Strides
      arrData, // Data pointer
      freeWhenDone); // Numpy array references this parent

  delete[] buffData;

  return arr;
}
