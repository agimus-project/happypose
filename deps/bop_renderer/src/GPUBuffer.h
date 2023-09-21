#pragma once

#include <string>
#include "glutils/gl_core_3_3.h"
#include "glutils/GLSLProgram.h"
#include "glutils/FrameBufferObject.h"

#include "glm/glm.hpp"
#include "Geometry.h"

enum ObjectAttribute {
  OA_LOCALPOS,
  OA_NORMALS,
  OA_COLORS,
  OA_TEXTURED,
  OA_DEPTH,
  OA_SEG
};

class GPUBuffer {
public:
  GPUBuffer();

  ~GPUBuffer();

  void initMesh(Geometry *mesh);

  void releaseMesh();

  void render(const glm::mat4 &modelTrans, const glm::mat4 &viewTrans,
              float fx, float fy, float cx, float cy,
              float skew, float xZero, float yZero,
              glm::vec3 lightCamPos, glm::vec3 lightColor,
              float lightAmbientWeight, float lightDiffuseWeight,
              float lightSpecularWeight, float lightSpecularShininess,
              bool useUniformColor, glm::vec3 uniformColor);

  void getPixelData(ObjectAttribute attr, float *data);

  bool initResources(int width, int height, std::string &errorString);

  void releaseResources();

private:
  GLSLProgram renderProg;
  FrameBufferObject fbo;

  Geometry *mesh;
  std::vector <GLuint> vbos;
  GLuint vao;
  GLuint texture;

  glm::mat4 calculateProjectionMatrix(
      float fx, float fy, float cx, float cy, float skew,
      float xZero, float yZero, const glm::mat4 &mv);
};
