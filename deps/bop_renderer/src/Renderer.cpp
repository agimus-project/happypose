#include <iostream>
#include <string>
#include <fstream>

#include "glutils/gl_core_3_3.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "PlyLoader.h"
#include "Renderer.h"


Renderer::Renderer() :
    width(0), height(0), fovy(0.0f),
    lightCamPos(glm::vec3(0.0f, 0.0f, 0.0f)),
    lightColor(glm::vec3(1.0f, 1.0f, 1.0f)),
    lightAmbientWeight(0.5f),
    lightDiffuseWeight(1.0f),
    lightSpecularWeight(0.0f),
    lightSpecularShininess(0.0f) {
}

Renderer::~Renderer() {
  deinit();
}

void Renderer::setLight(glm::vec3 lightCamPos, glm::vec3 lightColor,
                        float lightAmbientWeight, float lightDiffuseWeight,
                        float lightSpecularWeight, float lightSpecularShininess) {
  // OpenCV -> OpenGL coordinate system (flipping Y and Z axis)
  glm::vec3 lightCamPosGl = glm::vec3(
      lightCamPos[0], -lightCamPos[1], -lightCamPos[2]);

  this->lightCamPos = lightCamPosGl;
  this->lightColor = lightColor;
  this->lightAmbientWeight = lightAmbientWeight;
  this->lightDiffuseWeight = lightDiffuseWeight;
  this->lightSpecularWeight = lightSpecularWeight;
  this->lightSpecularShininess = lightSpecularShininess;
}

bool Renderer::init(int width, int height) {
  this->width = width;
  this->height = height;

  if (!context.init(width, height)) {
    std::cerr << "Error initializing GLFW: " << context.getError() << std::endl;
    return false;
  }

  // Init openGL function pointers
  context.makeCurrent();
  if (ogl_LoadFunctions() != ogl_LOAD_SUCCEEDED) {
    std::cerr << "Error initializing OpenGL functions" << std::endl;
    return false;
  }

  return true;
}

void Renderer::deinit() {
  for (auto &buf: renderers) {
    buf.second->releaseMesh();
    buf.second->releaseResources();
    delete buf.second;
  }
  renderers.clear();
  fileToHandle.clear();

  context.deinit();
}

int Renderer::getWidth() {
  return width;
}

int Renderer::getHeight() {
  return height;
}

bool Renderer::addObject(unsigned int objId, const std::string &filename) {
  // Load ply
  PlyLoader loader;
  if (!loader.loadFile(filename)) {
    std::cerr << "Error loading file " << filename << ": "
              << loader.getError() << std::endl;
    return false;
  }

  GPUBuffer *obj = new GPUBuffer();

  // Initialize renderer resources
  std::string rendererError;
  if (!obj->initResources(width, height, rendererError)) {
    std::cerr << "Error starting renderer: " << rendererError << std::endl;
    return false;
  }
  obj->initMesh(loader.getGeometry());

  renderers[objId] = obj;
  fileToHandle[filename] = objId;

  return true;
}

void Renderer::removeObject(unsigned int handle) {
  auto iter = renderers.find(handle);
  if (iter == renderers.end())
    return;

  context.makeCurrent();

  iter->second->releaseMesh();
  delete iter->second;
  renderers.erase(iter);

  // remove element from lookup maps as well
  for (auto iter = fileToHandle.begin(); iter != fileToHandle.end(); ++iter)
    if (iter->second == handle) {
      fileToHandle.erase(iter);
      break;
    }
}

void Renderer::renderObject(
    unsigned int handle, Pose &pose, float fx, float fy, float cx, float cy,
    float skew, float xZero, float yZero, bool useUniformColor,
    glm::vec3 uniformColor) {
  auto iter = renderers.find(handle);
  if (iter == renderers.end()) {
    std::cerr << "Error: Renderer not found." << std::endl;
    return;
  }

  context.makeCurrent();
  glm::mat4 modelTrans = glm::mat4();
  glm::mat4 viewTrans = pose.getPoseHomogen();
  iter->second->render(modelTrans, viewTrans, fx, fy, cx, cy,
                       skew, xZero, yZero, lightCamPos, lightColor,
                       lightAmbientWeight, lightDiffuseWeight,
                       lightSpecularWeight, lightSpecularShininess,
                       useUniformColor, uniformColor);
}

void Renderer::getBuffer(unsigned int handle, ObjectAttribute attr, float *data) {
  auto iter = renderers.find(handle);
  if (iter == renderers.end())
    return;

  context.makeCurrent();
  iter->second->getPixelData(attr, data);
}

float *Renderer::getBuffer(unsigned int handle, ObjectAttribute attr) {
  int numComp = 3;
  if (attr == OA_DEPTH) {
    numComp = 1;
  }
  float *data = new float[width * height * numComp];
  getBuffer(handle, attr, data);
  return data;
}
