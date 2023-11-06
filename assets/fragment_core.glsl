#version 330 core
out vec4 FragColor;

// in vec3 ourColor; //Attribute received from vertex_core shader
in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float mixValue;

void main(){
    // FragColor = vec4(1.0f, 0.2f, 0.6f, 1.0f);
    // FragColor = vec4(ourColor, 1.0);
    // FragColor = vec4(ourColor, 1.0f) * texture(texture1, TexCoord);
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), mixValue);
}