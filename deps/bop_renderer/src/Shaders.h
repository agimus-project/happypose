#pragma once

static string vertexShaderCode = \
"  #version 330                                                             \n"\
"                                                                           \n"\
"  // Attributes                                                            \n"\
"  layout(location = 0) in vec3 aPos;                                       \n"\
"  layout(location = 1) in vec3 aNormal;                                    \n"\
"  layout(location = 2) in vec3 aColor;                                     \n"\
"  layout(location = 3) in vec2 aTexcoord;                                  \n"\
"                                                                           \n"\
"  // Uniform variables                                                     \n"\
"  uniform mat4 uMV;                                                        \n"\
"  uniform mat4 uProj;                                                      \n"\
"  uniform mat4 uMVP;                                                       \n"\
"  uniform mat4 uNM;                                                        \n"\
"  uniform vec3 uLightCamPos;                                               \n"\
"                                                                           \n"\
"  // Varying variables                                                     \n"\
"  out vec3 vPos;                                                           \n"\
"  out vec3 vPosCam;                                                        \n"\
"  out vec3 vNormal;                                                        \n"\
"  out vec3 vNormalCam;                                                     \n"\
"  out vec3 vColor;                                                         \n"\
"  out vec2 vTexcoord;                                                      \n"\
"  out float vDepth;                                                        \n"\
"  out vec3 vLight;                                                         \n"\
"  out vec3 vView;                                                          \n"\
"  out vec3 vReflect;                                                       \n"\
"                                                                           \n"\
"  void main()                                                              \n"\
"  {                                                                        \n"\
"      vec4 posCam = uMV * vec4(aPos, 1.0);                                 \n"\
"                                                                           \n"\
"      vPos = aPos;                                                         \n"\
"      vPosCam = posCam.xyz;                                                \n"\
"      vNormal = aNormal;                                                   \n"\
"      vColor = aColor;                                                     \n"\
"      vTexcoord = aTexcoord;                                               \n"\
"      vDepth = -posCam.z;                                                  \n"\
"                                                                           \n"\
"      // Normal in camera coordinates                                      \n"\
"      vNormalCam = normalize((uNM * vec4(aNormal, 1.0)).xyz);              \n"\
"                                                                           \n"\
"      // Vector to the light                                               \n"\
"      vLight = normalize(uLightCamPos - posCam.xyz);                       \n"\
"                                                                           \n"\
"      // Vector to the camera                                              \n"\
"      vView = normalize(-vPosCam);                                         \n"\
"                                                                           \n"\
"      // Reflection vector                                                 \n"\
"      vReflect = normalize(reflect(-vLight, vNormalCam));                  \n"\
"                                                                           \n"\
"      gl_Position = uProj * posCam;                                        \n"\
"  }                                                                          ";

static string fragmentShaderCode = \
"  #version 330                                                             \n"\
"                                                                           \n"\
"  #define OA_LOCALPOS 0                                                    \n"\
"  #define OA_NORMALS 1                                                     \n"\
"  #define OA_COLORS 2                                                      \n"\
"  #define OA_TEXTURED 3                                                    \n"\
"  #define OA_DEPTH 4                                                       \n"\
"                                                                           \n"\
"  // Uniform variables                                                     \n"\
"  uniform sampler2D uTexture;                                              \n"\
"  uniform bool uUseFlatShading;                                            \n"\
"  uniform bool uUseTexture;                                                \n"\
"  uniform bool uUseUniformColor;                                           \n"\
"  uniform vec3 uUniformColor;                                              \n"\
"  uniform vec3 uLightColor;                                                \n"\
"  uniform float uLightAmbientWeight;                                       \n"\
"  uniform float uLightDiffuseWeight;                                       \n"\
"  uniform float uLightSpecularWeight;                                      \n"\
"  uniform float uLightSpecularShininess;                                   \n"\
"                                                                           \n"\
"  // Varying variables                                                     \n"\
"  in vec3 vPos;                                                            \n"\
"  in vec3 vPosCam;                                                         \n"\
"  in vec3 vNormal;                                                         \n"\
"  in vec3 vNormalCam;                                                      \n"\
"  in vec3 vColor;                                                          \n"\
"  in vec2 vTexcoord;                                                       \n"\
"  in float vDepth;                                                         \n"\
"  in vec3 vLight;                                                          \n"\
"  in vec3 vView;                                                           \n"\
"  in vec3 vReflect;                                                        \n"\
"                                                                           \n"\
"  // Output variables                                                      \n"\
"  out vec4 outColor[5];                                                    \n"\
"                                                                           \n"\
"  void main()                                                              \n"\
"  {                                                                        \n"\
"      vec3 surfNormalCam;                                                  \n"\
"      if(uUseFlatShading) {                                                \n"\
"          surfNormalCam = -normalize(cross(dFdx(vPosCam), dFdy(vPosCam))); \n"\
"      } else {                                                             \n"\
"          surfNormalCam = vNormalCam;                                      \n"\
"      }                                                                    \n"\
"                                                                           \n"\
"      float lightDiffuseFactor = max(                                      \n"\
"        dot(normalize(vLight), normalize(surfNormalCam)), 0.0);            \n"\
"                                                                           \n"\
"      float lightSpecularFactor = pow(                                     \n"\
"          max(dot(normalize(vView), normalize(vReflect)), 0.0),            \n"\
"          uLightSpecularShininess);                                        \n"\
"                                                                           \n"\
"      float lightFactor =                                                  \n"\
"          uLightAmbientWeight +                                            \n"\
"          uLightDiffuseWeight * lightDiffuseFactor +                       \n"\
"          uLightSpecularWeight * lightSpecularFactor;                      \n"\
"      //lightFactor = min(lightFactor, 1.0);                               \n"\
"      vec3 surfColor;                                                      \n"\
"      if(uUseTexture) {                                                    \n"\
"           surfColor = vec3(texture(                                       \n"\
"               uTexture, vec2(vTexcoord.x, 1.0 - vTexcoord.y)));           \n"\
"      }                                                                    \n"\
"      else if(uUseUniformColor) {                                          \n"\
"           surfColor = uUniformColor;                                      \n"\
"      }                                                                    \n"\
"      else {                                                               \n"\
"           surfColor = vColor;                                             \n"\
"      }                                                                    \n"\
"      vec3 color = min(lightFactor * uLightColor * surfColor, 1.0);        \n"\
"                                                                           \n"\
"      // Write object coordinates                                          \n"\
"      outColor[OA_LOCALPOS] = vec4(vPos, 0.0);                             \n"\
"                                                                           \n"\
"      // Write object normals                                              \n"\
"      outColor[OA_NORMALS] = vec4(vNormal, 0.0);                           \n"\
"                                                                           \n"\
"      // Write colors                                                      \n"\
"      outColor[OA_COLORS] = vec4(color, 0.0);                              \n"\
"                                                                           \n"\
"      // Write texture                                                     \n"\
"      outColor[OA_TEXTURED] = texture(                                     \n"\
"        uTexture, vec2(vTexcoord.x, 1.0 - vTexcoord.y));                   \n"\
"                                                                           \n"\
"      // Write depth                                                       \n"\
"      outColor[OA_DEPTH] = vec4(vDepth);                                   \n"\
"  }                                                                          ";

static string fragmentShaderFlatCode = \
"  #version 330                                                             \n"\
"                                                                           \n"\
"  #define OA_LOCALPOS 0                                                    \n"\
"  #define OA_NORMALS 1                                                     \n"\
"  #define OA_COLORS 2                                                      \n"\
"  #define OA_TEXTURED 3                                                    \n"\
"  #define OA_DEPTH 4                                                       \n"\
"                                                                           \n"\
"  // Uniform variables                                                     \n"\
"  uniform sampler2D uTexture;                                              \n"\
"  uniform bool uUseTexture;                                                \n"\
"  uniform vec3 uLightColor;                                                \n"\
"  uniform float uLightAmbientWeight;                                       \n"\
"  uniform float uLightDiffuseWeight;                                       \n"\
"  uniform float uLightSpecularWeight;                                      \n"\
"  uniform float uLightSpecularShininess;                                   \n"\
"                                                                           \n"\
"  // Varying variables                                                     \n"\
"  in vec3 vPos;                                                            \n"\
"  in vec3 vPosCam;                                                         \n"\
"  in vec3 vNormal;                                                         \n"\
"  in vec3 vNormalCam;                                                      \n"\
"  in vec3 vColor;                                                          \n"\
"  in vec2 vTexcoord;                                                       \n"\
"  in float vDepth;                                                         \n"\
"  in vec3 vLight;                                                          \n"\
"  in vec3 vView;                                                           \n"\
"  in vec3 vReflect;                                                        \n"\
"                                                                           \n"\
"  // Output variables                                                      \n"\
"  out vec4 outColor[5];                                                    \n"\
"                                                                           \n"\
"  void main()                                                              \n"\
"  {                                                                        \n"\
"      // Face normal in the camera coordinates                             \n"\
"      vec3 faceNormalCam = -normalize(cross(dFdx(vPosCam), dFdy(vPosCam)));\n"\
"                                                                           \n"\
"      float lightDiffuseWeight = max(                                      \n"\
"        dot(normalize(vLight), faceNormalCam), 0.0);                       \n"\
"                                                                           \n"\
"      float lightWeight = uLightAmbientWeight + lightDiffuseWeight;        \n"\
"      lightWeight = min(lightWeight, 1.0);                                 \n"\
"      vec3 color = lightWeight * uLightColor * vColor;                     \n"\
"                                                                           \n"\
"      // Write object coordinates                                          \n"\
"      outColor[OA_LOCALPOS] = vec4(vPos, 0.0);                             \n"\
"                                                                           \n"\
"      // Write object normals                                              \n"\
"      outColor[OA_NORMALS] = vec4(vNormal, 0.0);                           \n"\
"                                                                           \n"\
"      // Write colors                                                      \n"\
"      outColor[OA_COLORS] = vec4(color, 0.0);                              \n"\
"                                                                           \n"\
"      // Write texture                                                     \n"\
"      outColor[OA_TEXTURED] = texture(                                     \n"\
"        uTexture, vec2(vTexcoord.x, 1.0 - vTexcoord.y));                   \n"\
"                                                                           \n"\
"      // Write depth                                                       \n"\
"      outColor[OA_DEPTH] = vec4(vDepth);                                   \n"\
"  }                                                                          ";
