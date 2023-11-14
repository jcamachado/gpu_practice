cmake_minimum_required(VERSION 3.0.0)
project(my_particle_system VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)

add_subdirectory(lib/assimp-5.3.1)

add_executable(my_particle_system 
    assets/lamp.fs
    assets/object.fs
    assets/object.vs

    assets/models/sphere/scene.gltf
    assets/models/m4a1/scene.gltf

    lib/glad/glad.h 
    lib/stb/stb_image.h 
    lib/stb.cpp 
    
    src/glad.c
    src/main.cpp 

    src/graphics/light.h
    src/graphics/light.cpp
    src/graphics/material.h
    src/graphics/material.cpp
    src/graphics/mesh.h
    src/graphics/mesh.cpp
    src/graphics/model.h
    src/graphics/model.cpp
    src/graphics/shader.h
    src/graphics/shader.cpp
    src/graphics/texture.h
    src/graphics/texture.cpp

    src/graphics/models/cube.hpp
    src/graphics/models/lamp.hpp
    src/graphics/models/gun.hpp
    src/graphics/models/modelarray.hpp
    src/graphics/models/sphere.hpp

    src/io/camera.h
    src/io/camera.cpp
    src/io/joystick.h
    src/io/joystick.cpp
    src/io/keyboard.h
    src/io/keyboard.cpp
    src/io/mouse.h
    src/io/mouse.cpp
    src/io/screen.h
    src/io/screen.cpp

    src/physics/rigidbody.h
    src/physics/rigidbody.cpp
    src/physics/environment.h
    src/physics/environment.cpp
)


target_link_libraries(my_particle_system 
    glfw 
    GL
    assimp
)