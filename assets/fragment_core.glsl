#version 330 core
out vec4 FragColor;

in vec3 ourColor; //Attribute received from vertex_core shader

void main(){
    // FragColor = vec4(1.0f, 0.2f, 0.6f, 1.0f);
    FragColor = vec4(ourColor, 1.0);
}