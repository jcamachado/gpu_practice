/*
    We are seggregating the depth shader because in some cases, we only want the depth and not
    any color with it. So this module can be used alone without wasting any extra memory.
    Ex: Shadow mapping
    We dont need to know colors of an object. only how far it is from the light source.
*/


#version 330 core
out vec4 FragColor;

void main(){
    FragColor = vec4(1.0f);
}