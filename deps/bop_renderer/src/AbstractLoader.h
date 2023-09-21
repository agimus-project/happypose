#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "Geometry.h"

struct PreGeometry : public Geometry {
  virtual ~PreGeometry() {};
  std::vector <std::vector<int>> faceList;
  std::vector <std::vector<glm::vec2>> texcoordList;
};


class AbstractLoader {
protected:
  PreGeometry *geo;

public:
  virtual ~AbstractLoader() {};

  virtual bool loadFile(std::string filename) = 0;

  Geometry *getGeometry();

  const std::string &getError();

protected:
  std::string filename;
  std::string errorString;

  void makeGeometryIndexList();

  void calculateNormals();

  void faceTexCoordsToVertexTexCoords();

  bool loadTexture();

};
