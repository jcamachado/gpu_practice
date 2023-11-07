cmake_minimum_required(VERSION 3.0.0)
project(my_particle_system VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)

add_executable(my_particle_system 
    assets/object.fs
    assets/object.vs

    lib/glad/glad.h 
    lib/stb/stb_image.h 
    lib/stb.cpp 
    
    src/glad.c
    src/main.cpp 

    src/graphics/shader.h
    src/graphics/shader.cpp
    src/graphics/texture.h
    src/graphics/texture.cpp
    src/graphics/mesh.h
    src/graphics/mesh.cpp
    src/graphics/model.h
    src/graphics/model.cpp

    src/graphics/models/cube.hpp

    src/io/keyboard.h
    src/io/keyboard.cpp
    src/io/mouse.h
    src/io/mouse.cpp
    src/io/joystick.h
    src/io/joystick.cpp
    src/io/camera.h
    src/io/camera.cpp
    src/io/screen.h
    src/io/screen.cpp
)


target_link_libraries(my_particle_system 
    glfw 
    GL
)