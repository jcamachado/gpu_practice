//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
layout (location = 0) in vec3 aPos; //the layout is the location of the vertex attribute in the VBO
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aOffset;
layout (location = 4) in vec3 aSize;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec3 pos = aPos * aSize + aOffset;

    vs_out.FragPos = vec3(model * vec4(pos, 1.0)); // Its position in the world
    vs_out.Normal = mat3(transpose(inverse(model))) * aNormal; //Normal in world space

    gl_Position = projection * view * vec4(vs_out.FragPos, 1.0); //Order Matters!
    vs_out.TexCoord = aTexCoord;
}