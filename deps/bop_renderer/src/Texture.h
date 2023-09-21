#pragma once

#include <string>

struct Texture {
  std::string filename;
  std::vector<unsigned char> data;
  unsigned int width, height;

  Texture() : width(0), height(

  0) {};
};
