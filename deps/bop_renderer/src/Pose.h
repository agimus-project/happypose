#pragma once

#include "glm/glm.hpp"


class Pose {
public:
  Pose();

  Pose(glm::mat3 R, glm::vec3 t, float score = 0.0f);

  ~Pose();

  void setPose(const glm::mat3 &R, const glm::vec3 &t, float score = 0);

  inline glm::mat3 getRotation() { return R; }

  inline glm::vec3 getTranslation() { return t; }

  glm::mat4 getPoseHomogen();

private:
  glm::mat3 R;
  glm::vec3 t;
  float score;
};
