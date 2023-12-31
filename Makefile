# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jaxe/Repositories/gpu_study/gpu_practice

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jaxe/Repositories/gpu_study/gpu_practice

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\" \"assimp-dev\" \"libassimp5.3.0\" \"libassimp5.3.0-dev\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components
.PHONY : list_install_components/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jaxe/Repositories/gpu_study/gpu_practice/CMakeFiles /home/jaxe/Repositories/gpu_study/gpu_practice//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jaxe/Repositories/gpu_study/gpu_practice/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named my_particle_system

# Build rule for target.
my_particle_system: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 my_particle_system
.PHONY : my_particle_system

# fast build rule for target.
my_particle_system/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/build
.PHONY : my_particle_system/fast

#=============================================================================
# Target rules for targets named uninstall

# Build rule for target.
uninstall: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 uninstall
.PHONY : uninstall

# fast build rule for target.
uninstall/fast:
	$(MAKE) $(MAKESILENT) -f lib/assimp-5.3.1/CMakeFiles/uninstall.dir/build.make lib/assimp-5.3.1/CMakeFiles/uninstall.dir/build
.PHONY : uninstall/fast

#=============================================================================
# Target rules for targets named zlibstatic

# Build rule for target.
zlibstatic: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 zlibstatic
.PHONY : zlibstatic

# fast build rule for target.
zlibstatic/fast:
	$(MAKE) $(MAKESILENT) -f lib/assimp-5.3.1/contrib/zlib/CMakeFiles/zlibstatic.dir/build.make lib/assimp-5.3.1/contrib/zlib/CMakeFiles/zlibstatic.dir/build
.PHONY : zlibstatic/fast

#=============================================================================
# Target rules for targets named assimp

# Build rule for target.
assimp: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 assimp
.PHONY : assimp

# fast build rule for target.
assimp/fast:
	$(MAKE) $(MAKESILENT) -f lib/assimp-5.3.1/code/CMakeFiles/assimp.dir/build.make lib/assimp-5.3.1/code/CMakeFiles/assimp.dir/build
.PHONY : assimp/fast

#=============================================================================
# Target rules for targets named unit

# Build rule for target.
unit: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 unit
.PHONY : unit

# fast build rule for target.
unit/fast:
	$(MAKE) $(MAKESILENT) -f lib/assimp-5.3.1/test/CMakeFiles/unit.dir/build.make lib/assimp-5.3.1/test/CMakeFiles/unit.dir/build
.PHONY : unit/fast

lib/stb.o: lib/stb.cpp.o
.PHONY : lib/stb.o

# target to build an object file
lib/stb.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/lib/stb.cpp.o
.PHONY : lib/stb.cpp.o

lib/stb.i: lib/stb.cpp.i
.PHONY : lib/stb.i

# target to preprocess a source file
lib/stb.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/lib/stb.cpp.i
.PHONY : lib/stb.cpp.i

lib/stb.s: lib/stb.cpp.s
.PHONY : lib/stb.s

# target to generate assembly for a file
lib/stb.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/lib/stb.cpp.s
.PHONY : lib/stb.cpp.s

src/algorithms/bounds.o: src/algorithms/bounds.cpp.o
.PHONY : src/algorithms/bounds.o

# target to build an object file
src/algorithms/bounds.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o
.PHONY : src/algorithms/bounds.cpp.o

src/algorithms/bounds.i: src/algorithms/bounds.cpp.i
.PHONY : src/algorithms/bounds.i

# target to preprocess a source file
src/algorithms/bounds.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.i
.PHONY : src/algorithms/bounds.cpp.i

src/algorithms/bounds.s: src/algorithms/bounds.cpp.s
.PHONY : src/algorithms/bounds.s

# target to generate assembly for a file
src/algorithms/bounds.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.s
.PHONY : src/algorithms/bounds.cpp.s

src/algorithms/octree.o: src/algorithms/octree.cpp.o
.PHONY : src/algorithms/octree.o

# target to build an object file
src/algorithms/octree.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/algorithms/octree.cpp.o
.PHONY : src/algorithms/octree.cpp.o

src/algorithms/octree.i: src/algorithms/octree.cpp.i
.PHONY : src/algorithms/octree.i

# target to preprocess a source file
src/algorithms/octree.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/algorithms/octree.cpp.i
.PHONY : src/algorithms/octree.cpp.i

src/algorithms/octree.s: src/algorithms/octree.cpp.s
.PHONY : src/algorithms/octree.s

# target to generate assembly for a file
src/algorithms/octree.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/algorithms/octree.cpp.s
.PHONY : src/algorithms/octree.cpp.s

src/glad.o: src/glad.c.o
.PHONY : src/glad.o

# target to build an object file
src/glad.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/glad.c.o
.PHONY : src/glad.c.o

src/glad.i: src/glad.c.i
.PHONY : src/glad.i

# target to preprocess a source file
src/glad.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/glad.c.i
.PHONY : src/glad.c.i

src/glad.s: src/glad.c.s
.PHONY : src/glad.s

# target to generate assembly for a file
src/glad.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/glad.c.s
.PHONY : src/glad.c.s

src/graphics/light.o: src/graphics/light.cpp.o
.PHONY : src/graphics/light.o

# target to build an object file
src/graphics/light.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o
.PHONY : src/graphics/light.cpp.o

src/graphics/light.i: src/graphics/light.cpp.i
.PHONY : src/graphics/light.i

# target to preprocess a source file
src/graphics/light.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.i
.PHONY : src/graphics/light.cpp.i

src/graphics/light.s: src/graphics/light.cpp.s
.PHONY : src/graphics/light.s

# target to generate assembly for a file
src/graphics/light.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.s
.PHONY : src/graphics/light.cpp.s

src/graphics/material.o: src/graphics/material.cpp.o
.PHONY : src/graphics/material.o

# target to build an object file
src/graphics/material.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o
.PHONY : src/graphics/material.cpp.o

src/graphics/material.i: src/graphics/material.cpp.i
.PHONY : src/graphics/material.i

# target to preprocess a source file
src/graphics/material.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.i
.PHONY : src/graphics/material.cpp.i

src/graphics/material.s: src/graphics/material.cpp.s
.PHONY : src/graphics/material.s

# target to generate assembly for a file
src/graphics/material.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.s
.PHONY : src/graphics/material.cpp.s

src/graphics/mesh.o: src/graphics/mesh.cpp.o
.PHONY : src/graphics/mesh.o

# target to build an object file
src/graphics/mesh.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o
.PHONY : src/graphics/mesh.cpp.o

src/graphics/mesh.i: src/graphics/mesh.cpp.i
.PHONY : src/graphics/mesh.i

# target to preprocess a source file
src/graphics/mesh.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.i
.PHONY : src/graphics/mesh.cpp.i

src/graphics/mesh.s: src/graphics/mesh.cpp.s
.PHONY : src/graphics/mesh.s

# target to generate assembly for a file
src/graphics/mesh.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.s
.PHONY : src/graphics/mesh.cpp.s

src/graphics/model.o: src/graphics/model.cpp.o
.PHONY : src/graphics/model.o

# target to build an object file
src/graphics/model.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o
.PHONY : src/graphics/model.cpp.o

src/graphics/model.i: src/graphics/model.cpp.i
.PHONY : src/graphics/model.i

# target to preprocess a source file
src/graphics/model.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.i
.PHONY : src/graphics/model.cpp.i

src/graphics/model.s: src/graphics/model.cpp.s
.PHONY : src/graphics/model.s

# target to generate assembly for a file
src/graphics/model.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.s
.PHONY : src/graphics/model.cpp.s

src/graphics/shader.o: src/graphics/shader.cpp.o
.PHONY : src/graphics/shader.o

# target to build an object file
src/graphics/shader.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o
.PHONY : src/graphics/shader.cpp.o

src/graphics/shader.i: src/graphics/shader.cpp.i
.PHONY : src/graphics/shader.i

# target to preprocess a source file
src/graphics/shader.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.i
.PHONY : src/graphics/shader.cpp.i

src/graphics/shader.s: src/graphics/shader.cpp.s
.PHONY : src/graphics/shader.s

# target to generate assembly for a file
src/graphics/shader.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.s
.PHONY : src/graphics/shader.cpp.s

src/graphics/texture.o: src/graphics/texture.cpp.o
.PHONY : src/graphics/texture.o

# target to build an object file
src/graphics/texture.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o
.PHONY : src/graphics/texture.cpp.o

src/graphics/texture.i: src/graphics/texture.cpp.i
.PHONY : src/graphics/texture.i

# target to preprocess a source file
src/graphics/texture.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.i
.PHONY : src/graphics/texture.cpp.i

src/graphics/texture.s: src/graphics/texture.cpp.s
.PHONY : src/graphics/texture.s

# target to generate assembly for a file
src/graphics/texture.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.s
.PHONY : src/graphics/texture.cpp.s

src/io/camera.o: src/io/camera.cpp.o
.PHONY : src/io/camera.o

# target to build an object file
src/io/camera.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o
.PHONY : src/io/camera.cpp.o

src/io/camera.i: src/io/camera.cpp.i
.PHONY : src/io/camera.i

# target to preprocess a source file
src/io/camera.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/camera.cpp.i
.PHONY : src/io/camera.cpp.i

src/io/camera.s: src/io/camera.cpp.s
.PHONY : src/io/camera.s

# target to generate assembly for a file
src/io/camera.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/camera.cpp.s
.PHONY : src/io/camera.cpp.s

src/io/joystick.o: src/io/joystick.cpp.o
.PHONY : src/io/joystick.o

# target to build an object file
src/io/joystick.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o
.PHONY : src/io/joystick.cpp.o

src/io/joystick.i: src/io/joystick.cpp.i
.PHONY : src/io/joystick.i

# target to preprocess a source file
src/io/joystick.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.i
.PHONY : src/io/joystick.cpp.i

src/io/joystick.s: src/io/joystick.cpp.s
.PHONY : src/io/joystick.s

# target to generate assembly for a file
src/io/joystick.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.s
.PHONY : src/io/joystick.cpp.s

src/io/keyboard.o: src/io/keyboard.cpp.o
.PHONY : src/io/keyboard.o

# target to build an object file
src/io/keyboard.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o
.PHONY : src/io/keyboard.cpp.o

src/io/keyboard.i: src/io/keyboard.cpp.i
.PHONY : src/io/keyboard.i

# target to preprocess a source file
src/io/keyboard.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.i
.PHONY : src/io/keyboard.cpp.i

src/io/keyboard.s: src/io/keyboard.cpp.s
.PHONY : src/io/keyboard.s

# target to generate assembly for a file
src/io/keyboard.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.s
.PHONY : src/io/keyboard.cpp.s

src/io/mouse.o: src/io/mouse.cpp.o
.PHONY : src/io/mouse.o

# target to build an object file
src/io/mouse.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o
.PHONY : src/io/mouse.cpp.o

src/io/mouse.i: src/io/mouse.cpp.i
.PHONY : src/io/mouse.i

# target to preprocess a source file
src/io/mouse.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.i
.PHONY : src/io/mouse.cpp.i

src/io/mouse.s: src/io/mouse.cpp.s
.PHONY : src/io/mouse.s

# target to generate assembly for a file
src/io/mouse.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.s
.PHONY : src/io/mouse.cpp.s

src/main.o: src/main.cpp.o
.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/physics/environment.o: src/physics/environment.cpp.o
.PHONY : src/physics/environment.o

# target to build an object file
src/physics/environment.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o
.PHONY : src/physics/environment.cpp.o

src/physics/environment.i: src/physics/environment.cpp.i
.PHONY : src/physics/environment.i

# target to preprocess a source file
src/physics/environment.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.i
.PHONY : src/physics/environment.cpp.i

src/physics/environment.s: src/physics/environment.cpp.s
.PHONY : src/physics/environment.s

# target to generate assembly for a file
src/physics/environment.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.s
.PHONY : src/physics/environment.cpp.s

src/physics/rigidbody.o: src/physics/rigidbody.cpp.o
.PHONY : src/physics/rigidbody.o

# target to build an object file
src/physics/rigidbody.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o
.PHONY : src/physics/rigidbody.cpp.o

src/physics/rigidbody.i: src/physics/rigidbody.cpp.i
.PHONY : src/physics/rigidbody.i

# target to preprocess a source file
src/physics/rigidbody.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.i
.PHONY : src/physics/rigidbody.cpp.i

src/physics/rigidbody.s: src/physics/rigidbody.cpp.s
.PHONY : src/physics/rigidbody.s

# target to generate assembly for a file
src/physics/rigidbody.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.s
.PHONY : src/physics/rigidbody.cpp.s

src/scene.o: src/scene.cpp.o
.PHONY : src/scene.o

# target to build an object file
src/scene.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/scene.cpp.o
.PHONY : src/scene.cpp.o

src/scene.i: src/scene.cpp.i
.PHONY : src/scene.i

# target to preprocess a source file
src/scene.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/scene.cpp.i
.PHONY : src/scene.cpp.i

src/scene.s: src/scene.cpp.s
.PHONY : src/scene.s

# target to generate assembly for a file
src/scene.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/my_particle_system.dir/build.make CMakeFiles/my_particle_system.dir/src/scene.cpp.s
.PHONY : src/scene.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... install"
	@echo "... install/local"
	@echo "... install/strip"
	@echo "... list_install_components"
	@echo "... rebuild_cache"
	@echo "... uninstall"
	@echo "... assimp"
	@echo "... my_particle_system"
	@echo "... unit"
	@echo "... zlibstatic"
	@echo "... lib/stb.o"
	@echo "... lib/stb.i"
	@echo "... lib/stb.s"
	@echo "... src/algorithms/bounds.o"
	@echo "... src/algorithms/bounds.i"
	@echo "... src/algorithms/bounds.s"
	@echo "... src/algorithms/octree.o"
	@echo "... src/algorithms/octree.i"
	@echo "... src/algorithms/octree.s"
	@echo "... src/glad.o"
	@echo "... src/glad.i"
	@echo "... src/glad.s"
	@echo "... src/graphics/light.o"
	@echo "... src/graphics/light.i"
	@echo "... src/graphics/light.s"
	@echo "... src/graphics/material.o"
	@echo "... src/graphics/material.i"
	@echo "... src/graphics/material.s"
	@echo "... src/graphics/mesh.o"
	@echo "... src/graphics/mesh.i"
	@echo "... src/graphics/mesh.s"
	@echo "... src/graphics/model.o"
	@echo "... src/graphics/model.i"
	@echo "... src/graphics/model.s"
	@echo "... src/graphics/shader.o"
	@echo "... src/graphics/shader.i"
	@echo "... src/graphics/shader.s"
	@echo "... src/graphics/texture.o"
	@echo "... src/graphics/texture.i"
	@echo "... src/graphics/texture.s"
	@echo "... src/io/camera.o"
	@echo "... src/io/camera.i"
	@echo "... src/io/camera.s"
	@echo "... src/io/joystick.o"
	@echo "... src/io/joystick.i"
	@echo "... src/io/joystick.s"
	@echo "... src/io/keyboard.o"
	@echo "... src/io/keyboard.i"
	@echo "... src/io/keyboard.s"
	@echo "... src/io/mouse.o"
	@echo "... src/io/mouse.i"
	@echo "... src/io/mouse.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/physics/environment.o"
	@echo "... src/physics/environment.i"
	@echo "... src/physics/environment.s"
	@echo "... src/physics/rigidbody.o"
	@echo "... src/physics/rigidbody.i"
	@echo "... src/physics/rigidbody.s"
	@echo "... src/scene.o"
	@echo "... src/scene.i"
	@echo "... src/scene.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

