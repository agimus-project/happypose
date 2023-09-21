#pragma once

#include "AbstractLoader.h"

extern "C" {
#include "rply/rply.h"
}

class PlyLoader : public AbstractLoader {
public:
  virtual ~PlyLoader();

  bool loadFile(std::string filename);

private:
  bool hasNormals, hasFaceTexture;

  static int vertex_cb(p_ply_argument arg);

  static int normal_cb(p_ply_argument arg);

  static int texture_cb(p_ply_argument arg);

  static int face_cb(p_ply_argument arg);

  static int face_texcoord_cb(p_ply_argument arg);

  static int color_cb(p_ply_argument arg);

  static void error_cb(p_ply ply, const char *message);
};
