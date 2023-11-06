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
    src/io/keyboard.h
    src/io/keyboard.cpp
    src/io/mouse.h
    src/io/mouse.cpp
    src/io/joystick.h
    src/io/joystick.cpp
    src/io/camera.h
    src/io/camera.cpp
)


target_link_libraries(my_particle_system 
    glfw 
    GL
)