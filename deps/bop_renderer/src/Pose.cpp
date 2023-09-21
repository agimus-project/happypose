
#include <iostream>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Pose.h"

Pose::Pose() {

};

Pose::Pose(glm::mat3 R, glm::vec3 t, float score) {
  setPose(R, t, score);
}

Pose::~Pose() {
}

void Pose::setPose(const glm::mat3 &R, const glm::vec3 &t, float score) {
  // Flip Y and Z axis (OpenGL <-> OpenCV coordinate system)
  glm::mat3 yzFlip = glm::mat3();
  yzFlip[1][1] = -1.0f;
  yzFlip[2][2] = -1.0f;
  glm::mat3 Rf = yzFlip * R;
  glm::vec3 tf = yzFlip * t;

  this->R = Rf;
  this->t = tf;
  this->score = score;
}

glm::mat4 Pose::getPoseHomogen() {
  glm::mat4 trans = glm::translate(glm::mat4(), t) * glm::mat4(R);
  return trans;
}
