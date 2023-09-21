#include "PlyLoader.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <string.h>


PlyLoader::~PlyLoader() {

}


int PlyLoader::face_cb(p_ply_argument arg) {
  PlyLoader *self;
  long length, valIndex, faceIndex;

  ply_get_argument_user_data(arg, reinterpret_cast<void **>(&self), nullptr);
  ply_get_argument_element(arg, nullptr, &faceIndex);

  ply_get_argument_property(arg, nullptr, &length, &valIndex);
  if (valIndex == -1) {
    self->geo->faceList[faceIndex].resize(length);
    return 1;
  }

  self->geo->faceList[faceIndex][valIndex] = static_cast<int>(ply_get_argument_value(arg));
  return 1;
}


int PlyLoader::vertex_cb(p_ply_argument arg) {
  PlyLoader *self;
  long coordIndex, vertIndex;
  ply_get_argument_user_data(arg, reinterpret_cast<void **>(&self), &coordIndex);
  ply_get_argument_element(arg, nullptr, &vertIndex);

  self->geo->vertices[vertIndex][coordIndex] = static_cast<float>(ply_get_argument_value(arg));

  return 1;
}


int PlyLoader::color_cb(p_ply_argument arg) {
  PlyLoader *self;
  long coordIndex, vertIndex;
  ply_get_argument_user_data(arg, reinterpret_cast<void **>(&self), &coordIndex);
  ply_get_argument_element(arg, nullptr, &vertIndex);

  self->geo->colors[vertIndex][coordIndex] = static_cast<float>(ply_get_argument_value(arg)) / 255.0f;

  return 1;
}


int PlyLoader::normal_cb(p_ply_argument arg) {
  PlyLoader *self;
  long coordIndex, vertIndex;
  ply_get_argument_user_data(arg, reinterpret_cast<void **>(&self), &coordIndex);
  self->hasNormals = true;
  ply_get_argument_element(arg, nullptr, &vertIndex);

  self->geo->normals[vertIndex][coordIndex] = static_cast<float>(ply_get_argument_value(arg));

  return 1;
}


int PlyLoader::texture_cb(p_ply_argument arg) {
  PlyLoader *self;
  long coordIndex, vertIndex;
  ply_get_argument_user_data(arg, reinterpret_cast<void **>(&self), &coordIndex);
  ply_get_argument_element(arg, nullptr, &vertIndex);
  self->geo->texcoords[vertIndex][coordIndex] = static_cast<float>(ply_get_argument_value(arg));

  return 1;
}


int PlyLoader::face_texcoord_cb(p_ply_argument arg) {
  PlyLoader *self;
  long length, valIndex, faceIndex;

  ply_get_argument_user_data(arg, reinterpret_cast<void **>(&self), nullptr);
  self->hasFaceTexture = true;

  ply_get_argument_element(arg, nullptr, &faceIndex);

  ply_get_argument_property(arg, nullptr, &length, &valIndex);
  if (valIndex == -1) {
    self->geo->texcoordList[faceIndex].resize(length / 2);
    return 1;
  }

  float val = static_cast<float>(ply_get_argument_value(arg));
  if (valIndex % 2 == 0)
    self->geo->texcoordList[faceIndex][valIndex / 2].x = val;
  else
    self->geo->texcoordList[faceIndex][valIndex / 2].y = val;

  return 1;

}


void PlyLoader::error_cb(p_ply ply, const char *message) {
  PlyLoader *self;
  long dummy;
  ply_get_ply_user_data(ply, reinterpret_cast<void **>(&self), &dummy);
  self->errorString += std::string(message) + "\n";
}


bool PlyLoader::loadFile(std::string filename) {
  this->filename = filename;

  p_ply ply = ply_open(filename.c_str(), error_cb, 0, this);
  if (ply == nullptr)
    return false;

  if (!ply_read_header(ply)) {
    ply_close(ply);
    return false;
  }

  geo = new PreGeometry();

  // Look for textures in the comment section - hackyhacky ply!
  const char *texIdentifier = "TextureFile";
  const char *comment = nullptr;
  while ((comment = ply_get_next_comment(ply, comment)) != nullptr) {
    std::string commentLine = comment;
    size_t texLocation = commentLine.find(texIdentifier);
    if (texLocation != std::string::npos) {
      geo->texture.filename = commentLine.substr(texLocation + strlen(texIdentifier) + 1);
    }
  }

  long numVerts = 0;
  long numFaces = 0;

  numVerts = ply_set_read_cb(ply, "vertex", "x", vertex_cb, this, 0);
  ply_set_read_cb(ply, "vertex", "y", vertex_cb, this, 1);
  ply_set_read_cb(ply, "vertex", "z", vertex_cb, this, 2);

  ply_set_read_cb(ply, "vertex", "red", color_cb, this, 0);
  ply_set_read_cb(ply, "vertex", "green", color_cb, this, 1);
  ply_set_read_cb(ply, "vertex", "blue", color_cb, this, 2);

  ply_set_read_cb(ply, "vertex", "nx", normal_cb, this, 0);
  ply_set_read_cb(ply, "vertex", "ny", normal_cb, this, 1);
  ply_set_read_cb(ply, "vertex", "nz", normal_cb, this, 2);

  if (!geo->texture.filename.empty()) {
    ply_set_read_cb(ply, "vertex", "texture_u", texture_cb, this, 0);
    ply_set_read_cb(ply, "vertex", "texture_v", texture_cb, this, 1);
  }

  geo->vertices.resize(numVerts);
  geo->colors.resize(numVerts, glm::vec3(0.5f));
  geo->normals.resize(numVerts);
  geo->texcoords.resize(numVerts);

  hasNormals = false;
  hasFaceTexture = false;

  numFaces = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, this, 0);
  ply_set_read_cb(ply, "face", "texcoord", face_texcoord_cb, this, 0);
  geo->faceList.resize(numFaces);
  geo->texcoordList.resize(numFaces);

  if (!ply_read(ply)) {
    errorString =
        ply_close(ply);
    return false;
  }

  ply_close(ply);

  if (geo->texcoordList[0].size() != geo->faceList[0].size())
    geo->texcoordList.clear();

  if (!geo->texture.filename.empty()) {
    //std::cout << "Loading Texture "<<geo->texture.filename << std::endl;
    if (!loadTexture())
      return false;
  }
  if (hasFaceTexture) {
    //std::cout << "Converting face texture coordinates to vertex coordinates" << std::endl;
    faceTexCoordsToVertexTexCoords();
  }
  if (geo->indexList.empty()) {
    //std::cout << "Generating index list" << std::endl;
    makeGeometryIndexList();
  }
  if (!hasNormals) {
    //std::cout << "Generating Normals" << std::endl;
    calculateNormals();
  }
  if (geo->colors.empty())
    geo->colors.resize(geo->vertices.size());

  geo->calculateAABB();

  return true;
}
