#include "AbstractLoader.h"

#include <iostream>
#include "lodepng/lodepng.h"


Geometry *AbstractLoader::getGeometry() {
  return geo;
}


void AbstractLoader::makeGeometryIndexList() {
  // We need at least 3*numFaces entries
  geo->indexList.reserve(geo->faceList.size() * 3);

  // Convert faces to index list
  for (auto iter = geo->faceList.begin(); iter != geo->faceList.end(); ++iter) {
    if (iter->size() < 3 || iter->size() > 4)
      continue;

    // For triangles just append them
    if (iter->size() == 3) {
      geo->indexList.insert(geo->indexList.end(), iter->begin(), iter->end());
      continue;
    }

    // For quads turn them into two triangles
    geo->indexList.push_back((*iter)[0]);
    geo->indexList.push_back((*iter)[1]);
    geo->indexList.push_back((*iter)[2]);

    geo->indexList.push_back((*iter)[0]);
    geo->indexList.push_back((*iter)[2]);
    geo->indexList.push_back((*iter)[3]);
  }
}


void AbstractLoader::calculateNormals() {
  geo->normals.resize(geo->vertices.size());
  std::vector<int> numNormals(geo->vertices.size(), 0);

  for (int i = 0; i < geo->faceList.size(); ++i) {
    std::vector<int> &face = geo->faceList[i];
    glm::vec3 &p1 = geo->vertices[face[0]];
    glm::vec3 &p2 = geo->vertices[face[1]];
    glm::vec3 &p3 = geo->vertices[face[2]];

    glm::vec3 normal = glm::normalize(glm::cross(p2 - p1, p3 - p1));
    for (int j = 0; j < face.size(); ++j) {
      geo->normals[face[j]] += normal;
      ++numNormals[face[j]];
    }
  }

  for (int i = 0; i < geo->normals.size(); ++i) {
    geo->normals[i] /= static_cast<float>(numNormals[i]);
  }
}


void AbstractLoader::faceTexCoordsToVertexTexCoords() {
  if (geo->texcoordList.size() != geo->faceList.size()) {
    std::cout << "Error: Tex coords per face do not match face indices." << std::endl;
    return;
  }

  geo->texcoords.resize(geo->vertices.size());

  // Keep book of all copies of vertices with differing texture coordinates
  std::vector <std::vector<int>> vertexAlternatives(geo->vertices.size());

  // Go through all faces and check if the associated texture coordinates
  // match the texture coordinates at that index.
  // If not, copy the vertex for the new texture coordinate.
  for (int fi = 0; fi < geo->faceList.size(); ++fi) {

    std::vector<int> &faceVertices = geo->faceList[fi];

    for (int i = 0; i < faceVertices.size(); ++i) {
      int &vertexId = faceVertices[i];
      std::vector<int> &sameVertices = vertexAlternatives[vertexId];
      const glm::vec2 &texcoord = geo->texcoordList[fi][i];

      // If we have not assigned any texture coordinate then we do not need a copy
      if (sameVertices.empty()) {
        geo->texcoords[vertexId] = texcoord;
        sameVertices.push_back(vertexId);
        continue;
      }
      // Otherwise check if one of our vertices at the same
      // position but with different tex coordinates matches this
      // texture coordinate
      int newIndex = -1;
      for (int vid: sameVertices)
        if (geo->texcoords[vid] == texcoord) {
          newIndex = vid;
          break;
        }
      // No matching entry found? Make a copy of all properties
      if (newIndex == -1) {
        geo->vertices.push_back(geo->vertices[vertexId]);
        geo->normals.push_back(geo->normals[vertexId]);
        geo->colors.push_back(geo->colors[vertexId]);
        // Add new texture coordinate
        geo->texcoords.push_back(texcoord);
        // ... and add the entry to the list of same vertices
        newIndex = geo->texcoords.size() - 1;
        sameVertices.push_back(newIndex);
      }
      // Alter the face to now reference to the new vertex
      vertexId = newIndex;
    }
  }
}


bool AbstractLoader::loadTexture() {
  // Textures are assumed to be relative to the loaded file
  // Extract the path from the filename
  std::string path = "";
  size_t pathEnd = filename.find_last_of('/');
  if (pathEnd == std::string::npos)
    pathEnd = filename.find_last_of('\\');
  if (pathEnd != std::string::npos) {
    path = filename.substr(0, pathEnd + 1);
  }

  // Generate the complete name
  geo->texture.filename = path + geo->texture.filename;

  // Load the texture from png
  unsigned int error = lodepng::decode(geo->texture.data, geo->texture.width, geo->texture.height,
                                       geo->texture.filename);

  if (error != 0) {
    errorString = lodepng_error_text(error);
    return false;
  }

  return true;
}


const std::string &AbstractLoader::getError() {
  return errorString;
}
