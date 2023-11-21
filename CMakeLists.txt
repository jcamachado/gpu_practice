cmake_minimum_required(VERSION 3.0.0)
project(my_particle_system VERSION 0.1.0)
cmake_policy(SET CMP0072 NEW)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenGL 3.3 REQUIRED)
find_package(glfw3 REQUIRED)

add_subdirectory(lib/assimp-5.3.1)
add_executable(my_particle_system
    assets/lamp.fs
    assets/object.fs
    assets/object.vs
    
    assets/instanced/box.fs
    assets/instanced/box.vs
    assets/instanced/instanced.vs

    assets/models/sphere/scene.gltf
    assets/models/m4a1/scene.gltf

    lib/stb.cpp 
    
    src/glad.c
    src/main.cpp 
    src/scene.h 
    src/scene.cpp 

    src/algorithms/bounds.h
    src/algorithms/bounds.cpp
    src/algorithms/list.hpp
    src/algorithms/octree.h
    src/algorithms/octree.cpp
    src/algorithms/trie.hpp

    src/graphics/light.h
    src/graphics/light.cpp
    src/graphics/glmemory.hpp
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

    src/graphics/models/box.hpp
    src/graphics/models/cube.hpp
    src/graphics/models/lamp.hpp
    src/graphics/models/gun.hpp
    src/graphics/models/sphere.hpp

    src/io/camera.h
    src/io/camera.cpp
    src/io/joystick.h
    src/io/joystick.cpp
    src/io/keyboard.h
    src/io/keyboard.cpp
    src/io/mouse.h
    src/io/mouse.cpp

    src/physics/rigidbody.h
    src/physics/rigidbody.cpp
    src/physics/environment.h
    src/physics/environment.cpp
)
#including the header files glad/glad.h and stb_image
target_include_directories(my_particle_system PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(my_particle_system
    glfw 
    GL
    assimp
)
