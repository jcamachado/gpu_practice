cmake_minimum_required(VERSION 3.0.0)
project(my_particle_system VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)

add_executable(my_particle_system 
    src/main.cpp 
    lib/glad/glad.h 
    src/glad.c
    src/shader.h
    src/shader.cpp
    assets/vertex_core.glsl
    assets/fragment_core.glsl
)


target_link_libraries(my_particle_system 
    glfw 
    GL
)