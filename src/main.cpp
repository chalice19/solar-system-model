// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: 24 Jul 2020
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: IGR201 Practical; OpenGL and Shaders (DO NOT distribute!)
//
// Copyright 2020-2022 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// constants
const static float kSizeSun = 1;
const static float kSizeEarth = 0.5;
const static float kSizeMoon = 0.25;
const static float kRadOrbitEarth = 10;
const static float kRadOrbitMoon = 2;

// Window parameters
GLFWwindow *g_window = nullptr;
// GPU objects
GLuint g_program = 0; // A GPU program contains at least a vertex shader and a fragment shader
GLuint g_earthTexID;
GLuint g_moonTexID;
GLuint g_sunTexID;

// Update any accessible variable based on the current time
void update(const float currentTimeInSec, const char *message) {
  std::cout << message << currentTimeInSec << std::endl;
}

float getCurrentTime() {
  return static_cast<float>(glfwGetTime());
}

// Basic camera model
class Camera {
public:
  inline float getFov() const { return m_fov; }
  inline void setFoV(const float f) { m_fov = f; }
  inline float getAspectRatio() const { return m_aspectRatio; }
  inline void setAspectRatio(const float a) { m_aspectRatio = a; }
  inline float getNear() const { return m_near; }
  inline void setNear(const float n) { m_near = n; }
  inline float getFar() const { return m_far; }
  inline void setFar(const float n) { m_far = n; }
  inline void setPosition(const glm::vec3 &p) { m_pos = p; }
  inline glm::vec3 getPosition() { return m_pos; }

  inline glm::mat4 computeViewMatrix() const {
    return glm::lookAt(m_pos, glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
  }

  // Returns the projection matrix stemming from the camera intrinsic parameter.
  inline glm::mat4 computeProjectionMatrix() const {
    return glm::perspective(glm::radians(m_fov), m_aspectRatio, m_near, m_far);
  }

private:
  glm::vec3 m_pos = glm::vec3(0, 0, 0);
  float m_fov = 45.f;        // Field of view, in degrees
  float m_aspectRatio = 1.f; // Ratio between the width and the height of the image
  float m_near = 0.1f; // Distance before which geometry is excluded from the rasterization process
  float m_far = 10.f; // Distance after which the geometry is excluded from the rasterization process
};
Camera g_camera;

class Mesh {
  public:

  // should properly set up the geometry buffer
  void init() {
    // TODO: add argument for g_vao if use several shapes

    // Generate vertex array object
#ifdef _MY_OPENGL_IS_33_
    glGenVertexArrays(1, &m_vao);
#else
    glCreateVertexArrays(1, &m_vao);
#endif
    glBindVertexArray(m_vao);

    // Generate a GPU buffer to store the positions of the vertices
    size_t vertexBufferSize = sizeof(float)*m_vertexPositions.size();
#ifdef _MY_OPENGL_IS_33_
    glGenBuffers(1, &m_posVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
    glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, m_vertexPositions.data(), GL_DYNAMIC_READ);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);
#else
    glCreateBuffers(1, &m_posVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
    glNamedBufferStorage(m_posVbo, vertexBufferSize, m_vertexPositions.data(), GL_DYNAMIC_STORAGE_BIT); // Create a data storage on the GPU and fill it from a CPU array
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);
#endif

    // Same for an index buffer object that stores the list of indices of the
    // triangles forming the mesh
    size_t indexBufferSize = sizeof(unsigned int)*m_triangleIndices.size();
#ifdef _MY_OPENGL_IS_33_
    glGenBuffers(1, &m_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferSize, m_triangleIndices.data(), GL_DYNAMIC_READ);
#else
    glCreateBuffers(1, &m_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glNamedBufferStorage(m_ibo, indexBufferSize, m_triangleIndices.data(), GL_DYNAMIC_STORAGE_BIT);
#endif

    // Same for colors
#ifdef _MY_OPENGL_IS_33_
    size_t NormalsBufferSize = sizeof(float) * m_vertexNormals.size();
    glGenBuffers(1, &m_normalVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_normalVbo);
    glBufferData(GL_ARRAY_BUFFER, NormalsBufferSize, m_vertexNormals.data(), GL_DYNAMIC_READ);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(1);
#endif

    // Now for Texture Coordinates
#ifdef _MY_OPENGL_IS_33_
    size_t TexCoordsBufferSize = sizeof(float) * m_vertexTexCoords.size();
    glGenBuffers(1, &m_texCoordVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_texCoordVbo);
    glBufferData(GL_ARRAY_BUFFER, TexCoordsBufferSize, m_vertexTexCoords.data(), GL_DYNAMIC_READ);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(2);
#endif

    glBindVertexArray(0); // deactivate the VAO for now, will be activated again when rendering
  }

  // should be called in the main rendering loop
  void render(){
    float time = getCurrentTime();
    auto rotation_angle = time * glm::radians(30.0f);
    // std::cout << "the time is " << time << std::endl;

    glBindVertexArray(m_vao);     // activate the VAO storing geometry
    glActiveTexture(GL_TEXTURE0);

    glUniform1f(glGetUniformLocation(g_program, "ambientStrength"), 0.4f);

    model_sun = glm::mat4( 1.0f );


    /* Earth ...*/
    /* Calculate the the model transformation matrix */
    auto earth_angle = glm::radians(23.5f);
    
    // no rotation
    // model_earth = glm::translate(model_sun, glm::vec3(kRadOrbitEarth, 0.0f, 0.0f));
    
    // rotate both around the sun and around itself with the same period
    // model_earth = glm::rotate(model_sun, rotation_angle, glm::vec3(0.0f, 0.0f, 1.0f));
    
    // rotate only around the sun
    model_earth = glm::translate(model_sun, glm::vec3(cos(rotation_angle) * 10, sin(rotation_angle) * 10, 0.0f));


    /* Moon */
    model_moon = glm::rotate(model_earth, rotation_angle * 4, glm::vec3(0.0f, 0.0f, 1.0f));
    model_moon = glm::translate(model_moon, glm::vec3(kRadOrbitMoon, 0.0f, 0.0f));
    model_moon = glm::scale(model_moon, glm::vec3(kSizeMoon));
    
    glUniformMatrix4fv(glGetUniformLocation(g_program, "model"), 1, GL_FALSE, glm::value_ptr(model_moon));
    // glUniform3f(glGetUniformLocation(g_program, "objectColor"), 0.48f, 0.56f, 0.63f);
    glBindTexture(GL_TEXTURE_2D, g_moonTexID);
    glDrawElements(GL_TRIANGLES, m_triangleIndices.size(), GL_UNSIGNED_INT, 0);


    /* ... Earth */
    /* Pass the model transformation matrix */
    /* Set the texture */
    /* Draw the Earth */
    model_earth = glm::rotate(model_earth, earth_angle, glm::vec3(0.0, 1.0, 0.0));
    model_earth = glm::rotate(model_earth, rotation_angle * 2, glm::vec3(0.0, 0.0, 1.0));
    model_earth = glm::scale(model_earth, glm::vec3(kSizeEarth));
    glUniformMatrix4fv(glGetUniformLocation(g_program, "model"), 1, GL_FALSE, glm::value_ptr(model_earth));
    // glUniform3f(glGetUniformLocation(g_program, "objectColor"), 0.01f, 0.36f, 0.17f);
    glBindTexture(GL_TEXTURE_2D, g_earthTexID);
    glDrawElements(GL_TRIANGLES, m_triangleIndices.size(), GL_UNSIGNED_INT, 0);


    /* Sun */
    /* Pass the ident model transformation matrix */
    /* Set the ambient coefficient without darkening */
    /* Set the texture */
    /* Draw the Sun */
    model_sun = glm::rotate(model_sun, rotation_angle / 5, glm::vec3(0.0, 0.0, 1.0));
    glUniformMatrix4fv(glGetUniformLocation(g_program, "model"), 1, GL_FALSE, glm::value_ptr(model_sun));
    glUniform1f(glGetUniformLocation(g_program, "ambientStrength"), 1.0f);
    // glUniform3f(glGetUniformLocation(g_program, "objectColor"), 1.0f, 0.95f, 0.8f);
    glBindTexture(GL_TEXTURE_2D, g_sunTexID);
    glDrawElements(GL_TRIANGLES, m_triangleIndices.size(), GL_UNSIGNED_INT, 0);
  }

  void setRadius(float r) {
    R = r;
  }

  void setResolution(size_t r) {
    resolution_theta = r / 2;
    resolution_phi = r;

    size_t size = (resolution_phi + 1) * (resolution_theta - 1) + 2;
    m_vertexPositions.resize(size * 3);
    m_vertexNormals.resize(size * 3);
    m_vertexTexCoords.resize(size * 2);
  }

  size_t getResolutionTheta() {
    return resolution_theta;
  }

  size_t getResolutionPhi() {
    return resolution_phi;
  }

  void addVertex(size_t index, float x, float y, float z, float phi, float theta) {
    addTextureCoords(index, phi, theta);
    index *= 3;

    m_vertexPositions[index] = x;
    m_vertexPositions[index + 1] = y;
    m_vertexPositions[index + 2] = z;

    m_vertexNormals[index] = x;
    m_vertexNormals[index + 1] = y;
    m_vertexNormals[index + 2] = z;
  }

  void addTextureCoords(size_t index, float phi, float theta) {
    index *= 2;
    m_vertexTexCoords[index] = phi / (2 * M_PI);
    m_vertexTexCoords[index + 1] = theta / M_PI;
  }

  void addIndices(unsigned int i1, unsigned int i2, unsigned int i3) {
    if (i1 == i2) {
      return;
    }
    if (i2 == i3) {
      return;
    }
    if (i3 == i1) {
      return;
    }

    m_triangleIndices.insert(m_triangleIndices.end(), {i1, i2, i3});
  }

  size_t getGlobalIndex(size_t theta_i, size_t phi_i) {
    if (theta_i == 0) {
      return 0;
    }

    if (theta_i == resolution_theta) {
      return 1;
    }

    return phi_i * (resolution_theta - 1) + (theta_i - 1) + 2;
  }

  // should generate a unit sphere
  static std::shared_ptr<Mesh> genSphere(const size_t resolution = 16) {
    // update(static_cast<float>(glfwGetTime()), "start");

    auto sphere = std::make_shared<Mesh>();

    float R = 1;
    sphere->setRadius(R);
    sphere->setResolution(resolution);
    size_t resolution_theta = sphere->getResolutionTheta();
    size_t resolution_phi = sphere->getResolutionPhi();

    float theta_step = M_PI / resolution_theta;
    float phi_step = 2 * M_PI / resolution_phi;

    // Add upper pole point
    float x = 0;
    float y = 0;
    float z = R * cos(0);

    sphere->addVertex(0, x, y, z, 0, 0);

    // Add lower pole point
    x = 0;
    y = 0;
    z = R * cos(M_PI);

    sphere->addVertex(1, x, y, z, 0, M_PI);

    // add all other point except poles and link all
    for (size_t phi_i = 0; phi_i <= resolution_phi; ++phi_i) {
      for (size_t theta_i = 1; theta_i <= resolution_theta; ++theta_i) {
        float phi = phi_step * phi_i;
        float theta = theta_step * theta_i;
        size_t index = sphere->getGlobalIndex(theta_i, phi_i);

        x = R * sin(theta) * cos(phi);
        y = R * sin(theta) * sin(phi);
        z = R * cos(theta);

        if (theta_i != resolution) {
          sphere->addVertex(index, x, y, z, phi, theta);
        }

        if (phi_i != 0) {
          size_t i_lt = sphere->getGlobalIndex(theta_i - 1, phi_i - 1);
          size_t i_rt = sphere->getGlobalIndex(theta_i - 1, phi_i);
          size_t i_ld = sphere->getGlobalIndex(theta_i, phi_i - 1);

          sphere->addIndices(index, i_rt, i_lt);
          sphere->addIndices(i_lt, i_ld, index);
        }
      }
    }

    sphere->init();

    // update(static_cast<float>(glfwGetTime()), "end");
    return sphere;
  }

  private:
  std::vector<float> m_vertexPositions;           // [x0, y0, z0, x1, y1, z1, ...]
  std::vector<float> m_vertexNormals;             // 
  std::vector<unsigned int> m_triangleIndices;    // [tr0_0, tr0_1, tr0_2, tr1_0, tr1_1, tr1_2, ...] list of vertixes connected to a triangle
  std::vector<float> m_vertexTexCoords;
  
  GLuint m_vao = 0;
  GLuint m_posVbo = 0;
  GLuint m_normalVbo = 0;
  GLuint m_texCoordVbo = 0;
  GLuint m_ibo = 0;

  float R;
  size_t resolution_phi;
  size_t resolution_theta;

  glm::mat4 model_sun;
  glm::mat4 model_earth;
  glm::mat4 model_moon;
};

GLuint loadTextureFromFileToGPU(const std::string &filename) {
  int width, height, numComponents;
  // Loading the image in CPU memory using stb_image
  unsigned char *data = stbi_load(
    filename.c_str(),
    &width, &height,
    &numComponents, // 1 for a 8 bit grey-scale image, 3 for 24bits RGB image, 4 for 32bits RGBA image
    0);

  GLuint texID;
  // Create a texture and upload the image data in GPU memory
  glGenTextures(1, &texID); // generate an OpenGL texture container
  glBindTexture(GL_TEXTURE_2D, texID); // activate the texture
  // Setup the texture filtering option and repeat mode
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // Fill the GPU texture with the data stored in the CPU image
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

  // Free useless CPU memory
  stbi_image_free(data);
  glBindTexture(GL_TEXTURE_2D, 0); // unbind the texture

  return texID;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow* window, int width, int height) {
  g_camera.setAspectRatio(static_cast<float>(width)/static_cast<float>(height));
  glViewport(0, 0, (GLint)width, (GLint)height); // Dimension of the rendering region in the window
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if(action == GLFW_PRESS && key == GLFW_KEY_W) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  } else if(action == GLFW_PRESS && key == GLFW_KEY_F) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  } else if(action == GLFW_PRESS && (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)) {
    glfwSetWindowShouldClose(window, true); // Closes the application if the escape key is pressed
  }
}

void errorCallback(int error, const char *desc) {
  std::cout <<  "Error " << error << ": " << desc << std::endl;
}

void initGLFW() {
  glfwSetErrorCallback(errorCallback);

  // Initialize GLFW, the library responsible for window management
  if(!glfwInit()) {
    std::cerr << "ERROR: Failed to init GLFW" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Before creating the window, set some option flags
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create the window
  g_window = glfwCreateWindow(
    1024, 768,
    "Interactive 3D Applications (OpenGL) - Simple Solar System - Pilipyuk",
    nullptr, nullptr);
  if(!g_window) {
    std::cerr << "ERROR: Failed to open window" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  // Load the OpenGL context in the GLFW window using GLAD OpenGL wrangler
  glfwMakeContextCurrent(g_window);
  glfwSetWindowSizeCallback(g_window, windowSizeCallback);
  glfwSetKeyCallback(g_window, keyCallback);
}

void initOpenGL() {
  // Load extensions for modern OpenGL
  if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "ERROR: Failed to initialize OpenGL context" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  glCullFace(GL_BACK); // Specifies the faces to cull (here the ones pointing away from the camera)
  glEnable(GL_CULL_FACE); // Enables face culling (based on the orientation defined by the CW/CCW enumeration).
  glDepthFunc(GL_LESS);   // Specify the depth test for the z-buffer
  glEnable(GL_DEPTH_TEST);      // Enable the z-buffer test in the rasterization
  glClearColor(0.0f, 0.04f, 0.08f, 1.0f); // specify the background color, used any time the framebuffer is cleared
}

// Loads the content of an ASCII file in a standard C++ string
std::string file2String(const std::string &filename) {
  std::ifstream t(filename.c_str());
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

// Loads and compile a shader, before attaching it to a program
void loadShader(GLuint program, GLenum type, const std::string &shaderFilename) {
  GLuint shader = glCreateShader(type); // Create the shader, e.g., a vertex shader to be applied to every single vertex of a mesh
  std::string shaderSourceString = file2String(shaderFilename); // Loads the shader source from a file to a C++ string
  const GLchar *shaderSource = (const GLchar *)shaderSourceString.c_str(); // Interface the C++ string through a C pointer
  glShaderSource(shader, 1, &shaderSource, NULL); // load the vertex shader code
  glCompileShader(shader);
  GLint success;
  GLchar infoLog[512];
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if(!success) {
    glGetShaderInfoLog(shader, 512, NULL, infoLog);
    std::cout << "ERROR in compiling " << shaderFilename << "\n\t" << infoLog << std::endl;
  }
  glAttachShader(program, shader);
  glDeleteShader(shader);
}

void initGPUprogram() {
  g_program = glCreateProgram(); // Create a GPU program, i.e., two central shaders of the graphics pipeline
  loadShader(g_program, GL_VERTEX_SHADER, "vertexShader.glsl");
  loadShader(g_program, GL_FRAGMENT_SHADER, "fragmentShader.glsl");

  g_earthTexID = loadTextureFromFileToGPU("media/earth.jpg");
  g_moonTexID = loadTextureFromFileToGPU("media/moon.jpg");
  g_sunTexID = loadTextureFromFileToGPU("media/sun.jpg");
  glUniform1i(glGetUniformLocation(g_program, "material.albedoTex"), 0); // texture unit 0

  glLinkProgram(g_program); // The main GPU program is ready to be handle streams of polygons

  glUseProgram(g_program);
  // TODO: set shader variables, textures, etc.
}

void initCamera() {
  int width, height;
  glfwGetWindowSize(g_window, &width, &height);
  g_camera.setAspectRatio(static_cast<float>(width)/static_cast<float>(height));

  g_camera.setPosition(glm::vec3(0.0, -20.0, 15.0));
  g_camera.setNear(0.1);
  g_camera.setFar(80.1);
}

void init() {
  initGLFW();
  initOpenGL();
  initGPUprogram();
  initCamera();
}

void clear() {
  glDeleteProgram(g_program);

  glfwDestroyWindow(g_window);
  glfwTerminate();
}

// The main rendering call
void render() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Erase the color and z buffers.

  const glm::mat4 viewMatrix = g_camera.computeViewMatrix();
  const glm::mat4 projMatrix = g_camera.computeProjectionMatrix();
  const glm::vec3 camPosition = g_camera.getPosition();

  // compute the view matrix of the camera and pass it to the GPU program
  glUniformMatrix4fv(glGetUniformLocation(g_program, "viewMat"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
  // compute the projection matrix of the camera and pass it to the GPU program
  glUniformMatrix4fv(glGetUniformLocation(g_program, "projMat"), 1, GL_FALSE, glm::value_ptr(projMatrix));
  // pass camera position and pass it to the GPU program
  glUniform3f(glGetUniformLocation(g_program, "camPos"), camPosition[0], camPosition[1], camPosition[2]);
}

int main(int argc, char ** argv) {
  init(); // Your initialization code (user interface, OpenGL states, scene with geometry, material, lights, etc)
  auto sphere = Mesh::genSphere(100);
  while(!glfwWindowShouldClose(g_window)) {
    // update(static_cast<float>(glfwGetTime()));
    render();
    sphere->render();
    glfwSwapBuffers(g_window);
    glfwPollEvents();
  }
  clear();
  return EXIT_SUCCESS;
}