cmake_minimum_required(VERSION 3.0.0)
project(my_particle_system VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(Freetype REQUIRED VERSION 2.13.2)
add_subdirectory(lib/assimp-5.3.1)
add_subdirectory(lib/freetype-2.13.2)
add_executable(my_particle_system

    lib/glad/glad.h
    lib/jsoncpp/json.hpp
    lib/stb.cpp 
    lib/stb/stb_image.h
    
    src/glad.c
    src/main.cpp 
    src/scene.h 
    src/scene.cpp 

    src/algorithms/avl.h
    src/algorithms/avl.cpp
    src/algorithms/bounds.h
    src/algorithms/bounds.cpp
    src/algorithms/list.hpp
    src/algorithms/octree.h
    src/algorithms/octree.cpp
    src/algorithms/trie.hpp

    src/algorithms/cmathematics/cmathematics.h
    src/algorithms/cmathematics/cmathematics.cpp
    src/algorithms/cmathematics/matrix.h
    src/algorithms/cmathematics/matrix.cpp
    src/algorithms/cmathematics/vec.h
    src/algorithms/cmathematics/vec.cpp

    src/graphics/memory/vertexmemory.hpp
    src/graphics/memory/framememory.hpp
    src/graphics/memory/uniformmemory.hpp

    src/graphics/models/box.hpp
    src/graphics/models/brickwall.hpp
    src/graphics/models/cube.hpp
    src/graphics/models/lamp.hpp
    src/graphics/models/gun.hpp
    src/graphics/models/plane.hpp
    src/graphics/models/sphere.hpp

    src/graphics/objects/mesh.h
    src/graphics/objects/mesh.cpp
    src/graphics/objects/model.h
    src/graphics/objects/model.cpp

    src/graphics/rendering/cubemap.h
    src/graphics/rendering/cubemap.cpp
    src/graphics/rendering/light.h
    src/graphics/rendering/light.cpp
    src/graphics/rendering/material.h
    src/graphics/rendering/material.cpp
    src/graphics/rendering/shader.h
    src/graphics/rendering/shader.cpp
    src/graphics/rendering/text.h
    src/graphics/rendering/text.cpp
    src/graphics/rendering/texture.h
    src/graphics/rendering/texture.cpp

    src/io/camera.h
    src/io/camera.cpp
    src/io/joystick.h
    src/io/joystick.cpp
    src/io/keyboard.h
    src/io/keyboard.cpp
    src/io/mouse.h
    src/io/mouse.cpp

    src/physics/collisionmodel.h
    src/physics/collisionmodel.cpp
    src/physics/collisionmesh.h
    src/physics/collisionmesh.cpp
    src/physics/rigidbody.h
    src/physics/rigidbody.cpp
    src/physics/environment.h
    src/physics/environment.cpp
)
target_include_directories(my_particle_system PRIVATE ${FREETYPE_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/lib/jsoncpp)
#including the header files glad/glad.h and stb_image
# target_include_directories(my_particle_system PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(my_particle_system
    glfw 
    ${OPENGL_LIBRARIES}
    assimp    
    ${FREETYPE_LIBRARIES}
)
