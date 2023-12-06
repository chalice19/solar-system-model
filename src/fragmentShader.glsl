#version 330 core	     // Minimal GL version support expected from the GPU

out vec4 color;

uniform vec3 camPos;
// uniform vec3 objectColor;
uniform float ambientStrength;
in vec3 fPosition;
in vec3 fNormal;
in vec2 fTexCoords;

struct Material {
	sampler2D albedoTex; // texture unit, relate to glActivateTexture(GL_TEXTURE0 + i)
};

uniform Material material;

void main() {
	vec3 objectColor = texture(material.albedoTex, fTexCoords).rgb;

	vec3 n = normalize(fNormal);
	vec3 lPosition = vec3(0.0);
	vec3 l = normalize(lPosition - fPosition); 
	vec3 v = normalize(camPos - fPosition);
	vec3 r = dot(n, l) * 2.0 * n - l;

	float specularStrength = 0.4;
	// float ambientStrength = 0.7;
	// vec3 objectColor = vec3(0.95, 0.76, 0.2);
	// vec3 light_color = vec3(1.0);

	vec3 ambient = objectColor * ambientStrength;
	vec3 diffuse = max(dot(n, l), 0.0) * objectColor;
	vec3 specular = pow(max(dot(v, r), 0.0), 16) * objectColor * specularStrength;
	color = vec4(ambient + diffuse + specular, 1.0);
}

