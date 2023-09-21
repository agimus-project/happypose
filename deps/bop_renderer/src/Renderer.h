#pragma once

#include "GPUBuffer.h"
#include "Geometry.h"
#include "context/Context.h"
#include "Pose.h"

#include "math.h"
#include <vector>
#include <map>

class Renderer {
public:
  Renderer();

  ~Renderer();

  // Initialize the pose renderer with target width and height
  // Error messages are written to the console
  // Only call this once after startup
  bool init(int width, int height);

  // Uninitialize renderer
  // Call this once before exiting the application
  void deinit();

  int getWidth();

  int getHeight();

  void setLight(glm::vec3 lightCamPos, glm::vec3 lightColor,
                float lightAmbientWeight, float lightDiffuseWeight,
                float lightSpecularWeight, float lightSpecularShininess);

  // Load an object and add it to the list of objects.
  // You will need the returned handle to render poses and get the buffers
  // Alternatively, you can later retrieve the handle by the filename
  // using the method "byName"
  bool addObject(unsigned int objId, const std::string &filename);

  // Remove the object
  // Do this, when it is no longer needed
  void removeObject(unsigned int handle);

  void renderObject(unsigned int handle, Pose &pose, float fx, float fy,
                    float cx, float cy, float skew = 0.0f,
                    float xZero = 0.0f, float yZero = 0.0f,
                    bool useUniformColor = false,
                    glm::vec3 uniformColor = glm::vec3(1.0, 1.0, 1.0));

  // Get the contents of one attribute for the object identified by "handle".
  // Only returns useful values if "renderObject" was called before.
  // The array data musst be properly preallocated
  void getBuffer(unsigned int handle, ObjectAttribute attr, float *data);

  // The same as above, only that a buffer is allocated.
  // You need to remove the buffer yourself
  float *getBuffer(unsigned int handle, ObjectAttribute attr);

private:
  Context context;
  int width, height;
//    unsigned int handleCounter;
  float fovy;

  std::map<unsigned int, GPUBuffer *> renderers;
  std::map<std::string, unsigned int> fileToHandle;

  glm::vec3 lightCamPos;
  glm::vec3 lightColor;
  float lightAmbientWeight;
  float lightDiffuseWeight;
  float lightSpecularWeight;
  float lightSpecularShininess;
};
