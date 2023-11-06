cmake_minimum_required(VERSION 3.0.0)
project(my_particle_system VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)

add_executable(my_particle_system 
    src/main.cpp 
    assets/vertex_core.glsl
    assets/fragment_core.glsl
    lib/glad/glad.h 
    lib/stb/stb_image.h 
    lib/stb.cpp 
    src/glad.c
    src/shader.h
    src/shader.cpp
    src/io/Keyboard.h
    src/io/Keyboard.cpp
    src/io/Mouse.h
    src/io/Mouse.cpp
    src/io/Joystick.h
    src/io/Joystick.cpp
)


target_link_libraries(my_particle_system 
    glfw 
    GL
)