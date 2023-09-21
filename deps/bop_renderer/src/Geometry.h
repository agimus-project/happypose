#pragma once

#include <vector>
#include <string>

#include "Texture.h"

#include "glm/glm.hpp"

struct Geometry {
  // List of vertices
  std::vector <glm::vec3> vertices;
  // List of colors
  std::vector <glm::vec3> colors;
  // List of normals
  std::vector <glm::vec3> normals;
  // List of texture coordinates
  std::vector <glm::vec2> texcoords;
  // List of triangle indices
  std::vector<int> indexList;

  // Optional texture
  Texture texture;

  glm::vec3 boxMin, boxMax;

  void calculateAABB() {
    boxMin = glm::vec3(std::numeric_limits<float>::max());
    boxMax = -boxMin;
    for (auto iter = vertices.begin(); iter != vertices.end(); ++iter) {
      boxMin = glm::min(boxMin, *iter);
      boxMax = glm::max(boxMax, *iter);
    }
  }

  // Make virtual destructor to generate vtable during compilation
  virtual ~Geometry() {};
};
