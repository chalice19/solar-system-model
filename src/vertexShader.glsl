#version 330 core            // Minimal GL version support expected from the GPU

layout(location=0) in vec3 vPosition;
layout(location=1) in vec3 vNormal;
layout(location=2) in vec2 vTexCoords;

uniform mat4 viewMat, projMat, model;
out vec3 fNormal;
out vec3 fPosition;
out vec2 fTexCoords;

void main() {
        gl_Position = projMat * viewMat * model * vec4(vPosition, 1.0); // mandatory to rasterize 
        
        fPosition = vec3(model * vec4(vPosition, 1.0));
        fNormal = mat3(transpose(inverse(model))) * vNormal;
        fTexCoords = vTexCoords;
}

