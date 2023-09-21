#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Renderer.h"

namespace py = pybind11;


class PythonWrapper {
public:
  PythonWrapper();

  ~PythonWrapper();

  bool init(int width, int height);

  void setLight(std::array<float, 3> lightCamPos,
                std::array<float, 3> lightColor,
                float lightAmbientWeight, float lightDiffuseWeight,
                float lightSpecularWeight, float lightSpecularShininess);

  bool addObject(unsigned int objId, const std::string &filename);

  void removeObject(unsigned int handle);

  void renderObject(unsigned int handle,
                    std::array<float, 9> R,
                    std::array<float, 3> t,
                    float fx, float fy, float cx, float cy,
                    float skew, float xZero, float yZero,
                    bool useUniformColor,
                    float uniformColorR,
                    float uniformColorG,
                    float uniformColorB);

  py::array getDepthImage(unsigned int handle);

  py::array getColorImage(unsigned int handle);

  py::array getLocalPosImage(unsigned int handle);

private:
  Renderer renderer;
};


// Interface for Python.
PYBIND11_MODULE(bop_renderer, m) {
  py::class_<PythonWrapper>(m, "Renderer")
    .def (py::init())
    .def("init", &PythonWrapper::init)
    .def("set_light", &PythonWrapper::setLight)
    .def("add_object", &PythonWrapper::addObject)
    .def("remove_object", &PythonWrapper::removeObject)
    .def("render_object", &PythonWrapper::renderObject,
      py::arg("obj_id"),
      py::arg("R"),
      py::arg("t"),
      py::arg("fx"),
      py::arg("fy"),
      py::arg("cx"),
      py::arg("cy"),
      py::arg("skew")=0.0f,
      py::arg("x_xero")=0.0f,
      py::arg("y_zero")=0.0f,
      py::arg("use_uniform_color")=false,
      py::arg("uniform_color_r")=0.5,
      py::arg("uniform_color_g")=0.5,
      py::arg("uniform_color_b")=0.5
      )
    .def("get_depth_image", &PythonWrapper::getDepthImage)
    .def("get_color_image", &PythonWrapper::getColorImage)
    .def("get_local_pos_image", &PythonWrapper::getLocalPosImage);
}
