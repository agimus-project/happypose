#include <string>
#include "glm/gtc/matrix_transform.hpp"
#include "GPUBuffer.h"
#include "Shaders.h"

GPUBuffer::GPUBuffer() : vao(0), mesh(nullptr) {
}

GPUBuffer::~GPUBuffer() {
  if (mesh != nullptr) {
    delete mesh;
  }
}

void GPUBuffer::initMesh(Geometry *mesh) {
  this->mesh = mesh;

  // Create buffers
  glGenVertexArrays(1, &vao);

  vbos.resize(4);
  glGenBuffers(5, vbos.data());

  glBindVertexArray(vao);

  // Upload position
  glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
  glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size() * 3 * sizeof(float), mesh->vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  // Upload normals
  glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
  glBufferData(GL_ARRAY_BUFFER, mesh->normals.size() * 3 * sizeof(float), mesh->normals.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(1);

  // Upload colors
  glBindBuffer(GL_ARRAY_BUFFER, vbos[2]);
  glBufferData(GL_ARRAY_BUFFER, mesh->colors.size() * 3 * sizeof(float), mesh->colors.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(2);

  // Upload uv coordinates
  glBindBuffer(GL_ARRAY_BUFFER, vbos[3]);
  glBufferData(GL_ARRAY_BUFFER, mesh->texcoords.size() * 2 * sizeof(float), mesh->texcoords.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(3);

  // Upload face indices
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[4]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->indexList.size() * sizeof(int), mesh->indexList.data(), GL_STATIC_DRAW);

  // Load texture
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  if (!mesh->texture.data.empty()) {
    //std::cout << "Using externally saved texture." << std::endl;
    // Upload texture data if present
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mesh->texture.width, mesh->texture.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, mesh->texture.data.data());
  } else {
    //std::cout << "Using white color." << std::endl;
    // If no texture is given then upload a dummy texture consisting of 1 white pixel
    unsigned char whitePixel[4 * 4] = {255, 255, 255, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, whitePixel);
  }
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glGenerateMipmap(GL_TEXTURE_2D);
}

void GPUBuffer::render(const glm::mat4 &modelTrans, const glm::mat4 &viewTrans,
                       float fx, float fy, float cx, float cy,
                       float skew, float xZero, float yZero,
                       const glm::vec3 lightCamPos, const glm::vec3 lightColor,
                       float lightAmbientWeight, float lightDiffuseWeight,
                       float lightSpecularWeight, float lightSpecularShininess,
                       bool useUniformColor, glm::vec3 uniformColor) {
  // Enable everything
  fbo.enableFBO();
  renderProg.use();

  glBindVertexArray(vao);
  GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                           GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
                           GL_COLOR_ATTACHMENT4};
  glDrawBuffers(5, draw_buffers);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);

  // Model-View transformation
  glm::mat4 mvTrans = viewTrans * modelTrans;

  // Projection transformation
  glm::mat4 projTrans = calculateProjectionMatrix(
      fx, fy, cx, cy, skew, xZero, yZero, mvTrans);

  // Model-View-Projection transformation
  glm::mat4 mvpTrans = projTrans * mvTrans;

  // Normal matrix (Ref: http://www.songho.ca/opengl/gl_normaltransform.html)
  glm::mat4 nmTrans = glm::transpose(glm::inverse(mvTrans));

  bool useTexture = !mesh->texture.filename.empty();
  bool useFlatShading = false;

  // Set parameters
  renderProg.setUniform("uMV", mvTrans);
  renderProg.setUniform("uProj", projTrans);
  renderProg.setUniform("uMVP", mvpTrans);
  renderProg.setUniform("uNM", nmTrans);
  renderProg.setUniform("uTexture", 0);
  renderProg.setUniform("uUseTexture", useTexture);
  renderProg.setUniform("uUseFlatShading", useFlatShading);
  renderProg.setUniform("uUseUniformColor", useUniformColor);
  renderProg.setUniform("uUniformColor", uniformColor);
  renderProg.setUniform("uLightCamPos", lightCamPos);
  renderProg.setUniform("uLightColor", lightColor);
  renderProg.setUniform("uLightAmbientWeight", lightAmbientWeight);
  renderProg.setUniform("uLightDiffuseWeight", lightDiffuseWeight);
  renderProg.setUniform("uLightSpecularWeight", lightSpecularWeight);
  renderProg.setUniform("uLightSpecularShininess", lightSpecularShininess);

//    std::cout << "Model-View matrix:" << std::endl;
//    for(unsigned r = 0; r < 4; r++) {
//        for(unsigned c = 0; c < 4; c++) {
//            std::cout << mvTrans[r][c] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    std::cout << "Projection matrix:" << std::endl;
//    for(unsigned r = 0; r < 4; r++) {
//        for(unsigned c = 0; c < 4; c++) {
//            std::cout << projTrans[r][c] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    std::cout << "Model-View-Projection matrix:" << std::endl;
//    for(unsigned r = 0; r < 4; r++) {
//        for(unsigned c = 0; c < 4; c++) {
//            std::cout << mvpTrans[r][c] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    std::cout << "Normal matrix:" << std::endl;
//    for(unsigned r = 0; r < 4; r++) {
//        for(unsigned c = 0; c < 4; c++) {
//            std::cout << nmTrans[r][c] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

  // Rendering
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  // Do not cull faces because some object models have holes in the surface
  glDisable(GL_CULL_FACE);
  glDrawElements(GL_TRIANGLES, mesh->indexList.size(), GL_UNSIGNED_INT, nullptr);
}

glm::mat4 GPUBuffer::calculateProjectionMatrix(
    float fx, float fy, float cx, float cy,
    float skew, float xZero, float yZero, const glm::mat4 &mv) {

  // Conventionally estimate near and far clipping plane to get the
  // best z buffer precision.
  // This is done by transforming the bounding box of the mesh
  // and find the closest and farthest point
  float znear = std::numeric_limits<float>::max();
  float zfar = -znear;

  // Since we will only need the z value we extract the third row from the mv matrix
  glm::vec4 mvz(mv[0][2], mv[1][2], mv[2][2], mv[3][2]);
  for (int i = 0; i < 8; ++i) {
    // Construct point
    glm::vec4 bbPoint(
        (i & 1) ? mesh->boxMax.x : mesh->boxMin.x,
        (i & 2) ? mesh->boxMax.y : mesh->boxMin.y,
        (i & 4) ? mesh->boxMax.z : mesh->boxMin.z,
        1.0f);

    // get z-value by multiplying with z row from modelview
    float zVal = glm::dot(bbPoint, mvz);
    znear = glm::min(znear, -zVal);
    zfar = glm::max(zfar, -zVal);
  }

  const float distQuot = 0.001f;
  if (znear < zfar * distQuot)
    znear = zfar * distQuot;

  float depth = zfar - znear;
  float q = -(zfar + znear) / depth;
  float qn = -2.0f * (zfar * znear) / depth;

  auto w = static_cast<float>(fbo.getWidth());
  auto h = static_cast<float>(fbo.getHeight());

  glm::mat4 proj;

  //set first line
  proj[0][0] = 2.0f * fx / w;
  proj[1][0] = -2.0f * skew / w;
  proj[2][0] = (-2.0f * cx + w + 2.0f * xZero) / w;
  proj[3][0] = 0.0f;

  //set second line
  proj[0][1] = 0.0f;
  proj[1][1] = -2.0f * fy / h;
  proj[2][1] = (-2.0f * cy + h + 2.0f * yZero) / h;
  proj[3][1] = 0.0f;

  //set thrid line
  proj[0][2] = 0.0f;
  proj[1][2] = 0.0f;
  proj[2][2] = q;
  proj[3][2] = qn;

  //set fourth line
  proj[0][3] = 0.0f;
  proj[1][3] = 0.0f;
  proj[2][3] = -1.0f;
  proj[3][3] = 0.0f;

  return proj;
}

bool GPUBuffer::initResources(int width, int height, std::string &errorString) {

  // Initialize shaders
  // These variables are static to have a cheap way for a singleton,
  // so only initialize them if they are not already initialized
  //if (!renderProg.isLinked()) {
  try {
    renderProg.compileShader(vertexShaderCode, GLSLShader::VERTEX);
    renderProg.compileShader(fragmentShaderCode, GLSLShader::FRAGMENT);
    renderProg.link();
  } catch (std::exception &e) {
    errorString = e.what();
    return false;
  }
  //}

  //if (!fbo.isCreated()) {
  // Initialize frame buffer
  fbo.setDepthBufferEnabled(true);
  fbo.attachColorBuffer(width, height, GL_RGB32F); // object coordinates
  fbo.attachColorBuffer(width, height, GL_RGB32F); // normals
  fbo.attachColorBuffer(width, height, GL_RGB32F); // colors
  fbo.attachColorBuffer(width, height, GL_RGB32F); // textured
  fbo.attachColorBuffer(width, height, GL_R32F);   // depth

  if (!fbo.createFBO()) {
    errorString = "FrameBuffer object creation failed: ";
    errorString += fbo.getStatusString();
    fbo.destroyFBO();
    return false;
  }
  //}

  return true;
}

void GPUBuffer::releaseResources() {
  fbo.destroyFBO();
  renderProg.release();
}

void GPUBuffer::releaseMesh() {
  if (glIsVertexArray(vao)) {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(vbos.size(), vbos.data());

    glDeleteTextures(1, &texture);
  }

  if (mesh != nullptr) {
    delete mesh;
    mesh = nullptr;
  }
}

void GPUBuffer::getPixelData(ObjectAttribute attr, float *data) {

  if (attr != OA_DEPTH) {
    glBindTexture(GL_TEXTURE_2D, fbo.getColorBufferHandle(static_cast<int>(attr)));
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, data);
  } else {
    glBindTexture(GL_TEXTURE_2D, fbo.getColorBufferHandle(static_cast<int>(attr)));
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);
  }
}
