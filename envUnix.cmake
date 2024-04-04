
# Optional set TINYOBJ_PATH to target specific version, otherwise defaults to external/tinyobjloader
# Accepted version if the .h from https://github.com/tinyobjloader/tinyobjloader/blob/cc327eecf7f8f4139932aec8d75db2d091f412ef/tiny_obj_loader.h
# the header file is inside the folder tinyobjloader, thus we need too add the folder to the include path
set(TINYOBJ_PATH /usr/local/include/tinyobjloader)
# GLSLC = /usr/local/bin/glslc
# TINYOBJ_PATH = src/libs/tinyobjloader
# SHADER_DIR = $(CURDIR)/src/shaders
set(GLSLC /usr/local/bin/glslc)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)