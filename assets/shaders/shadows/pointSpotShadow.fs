#version 330 core

in vec4 FragPos;

uniform vec3 lightPos;
uniform float farPlane;

void main()
{   
    /*
        We will linearize the depth values before writing them to the depth map to simplify the calculations
        in object.fs
    */
    // Get distance between fragment and light source
    float lightDist = length(FragPos.xyz - lightPos);

    // Map to [0, 1] range by dividing by farPlane
    lightDist /= farPlane;

    // Write to depth map
    gl_FragDepth = lightDist;
}
