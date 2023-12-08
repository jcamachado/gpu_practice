#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D bufferTex;

float near = 0.1;
float far = 100.0;



void main(){
    // Depth Map
    // float depthValue = texture(bufferTex, TexCoord).r; 
    // float z = depthValue * 2.0 - 1.0; // Transform to Normalized Device Coordinates (NDC) [0, 1] -> [-1, 1]
    // float linearDepth = (2.0 * near * far) / (depthValue * (far - near) - (far + near)); 
    // float factor = (near + linearDepth) / (near - far); // Transform back to [0, 1] range

    // FragColor = vec4(vec3(1 - factor), 1.0);

    // Color Map
    FragColor = texture(bufferTex, TexCoord);
}