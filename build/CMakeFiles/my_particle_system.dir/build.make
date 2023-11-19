# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

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
CMAKE_BINARY_DIR = /home/jaxe/Repositories/gpu_study/gpu_practice/build

# Include any dependencies generated for this target.
include CMakeFiles/my_particle_system.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/my_particle_system.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/my_particle_system.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/my_particle_system.dir/flags.make

CMakeFiles/my_particle_system.dir/lib/stb.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/lib/stb.cpp.o: ../lib/stb.cpp
CMakeFiles/my_particle_system.dir/lib/stb.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/my_particle_system.dir/lib/stb.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/lib/stb.cpp.o -MF CMakeFiles/my_particle_system.dir/lib/stb.cpp.o.d -o CMakeFiles/my_particle_system.dir/lib/stb.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/lib/stb.cpp

CMakeFiles/my_particle_system.dir/lib/stb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/lib/stb.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/lib/stb.cpp > CMakeFiles/my_particle_system.dir/lib/stb.cpp.i

CMakeFiles/my_particle_system.dir/lib/stb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/lib/stb.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/lib/stb.cpp -o CMakeFiles/my_particle_system.dir/lib/stb.cpp.s

CMakeFiles/my_particle_system.dir/src/glad.c.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/glad.c.o: ../src/glad.c
CMakeFiles/my_particle_system.dir/src/glad.c.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/my_particle_system.dir/src/glad.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/glad.c.o -MF CMakeFiles/my_particle_system.dir/src/glad.c.o.d -o CMakeFiles/my_particle_system.dir/src/glad.c.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/glad.c

CMakeFiles/my_particle_system.dir/src/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/my_particle_system.dir/src/glad.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/glad.c > CMakeFiles/my_particle_system.dir/src/glad.c.i

CMakeFiles/my_particle_system.dir/src/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/my_particle_system.dir/src/glad.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/glad.c -o CMakeFiles/my_particle_system.dir/src/glad.c.s

CMakeFiles/my_particle_system.dir/src/main.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/my_particle_system.dir/src/main.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/my_particle_system.dir/src/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/main.cpp.o -MF CMakeFiles/my_particle_system.dir/src/main.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/main.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/main.cpp

CMakeFiles/my_particle_system.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/main.cpp > CMakeFiles/my_particle_system.dir/src/main.cpp.i

CMakeFiles/my_particle_system.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/main.cpp -o CMakeFiles/my_particle_system.dir/src/main.cpp.s

CMakeFiles/my_particle_system.dir/src/scene.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/scene.cpp.o: ../src/scene.cpp
CMakeFiles/my_particle_system.dir/src/scene.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/my_particle_system.dir/src/scene.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/scene.cpp.o -MF CMakeFiles/my_particle_system.dir/src/scene.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/scene.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/scene.cpp

CMakeFiles/my_particle_system.dir/src/scene.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/scene.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/scene.cpp > CMakeFiles/my_particle_system.dir/src/scene.cpp.i

CMakeFiles/my_particle_system.dir/src/scene.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/scene.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/scene.cpp -o CMakeFiles/my_particle_system.dir/src/scene.cpp.s

CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o: ../src/algorithms/bounds.cpp
CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o -MF CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/algorithms/bounds.cpp

CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/algorithms/bounds.cpp > CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.i

CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/algorithms/bounds.cpp -o CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.s

CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o: ../src/graphics/light.cpp
CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o -MF CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/light.cpp

CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/light.cpp > CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.i

CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/light.cpp -o CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.s

CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o: ../src/graphics/material.cpp
CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o -MF CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/material.cpp

CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/material.cpp > CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.i

CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/material.cpp -o CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.s

CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o: ../src/graphics/mesh.cpp
CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o -MF CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/mesh.cpp

CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/mesh.cpp > CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.i

CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/mesh.cpp -o CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.s

CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o: ../src/graphics/model.cpp
CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o -MF CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/model.cpp

CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/model.cpp > CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.i

CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/model.cpp -o CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.s

CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o: ../src/graphics/shader.cpp
CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o -MF CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/shader.cpp

CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/shader.cpp > CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.i

CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/shader.cpp -o CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.s

CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o: ../src/graphics/texture.cpp
CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o -MF CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/texture.cpp

CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/texture.cpp > CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.i

CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/graphics/texture.cpp -o CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.s

CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o: ../src/io/camera.cpp
CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o -MF CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/camera.cpp

CMakeFiles/my_particle_system.dir/src/io/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/io/camera.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/camera.cpp > CMakeFiles/my_particle_system.dir/src/io/camera.cpp.i

CMakeFiles/my_particle_system.dir/src/io/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/io/camera.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/camera.cpp -o CMakeFiles/my_particle_system.dir/src/io/camera.cpp.s

CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o: ../src/io/joystick.cpp
CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o -MF CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/joystick.cpp

CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/joystick.cpp > CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.i

CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/joystick.cpp -o CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.s

CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o: ../src/io/keyboard.cpp
CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o -MF CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/keyboard.cpp

CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/keyboard.cpp > CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.i

CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/keyboard.cpp -o CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.s

CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o: ../src/io/mouse.cpp
CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o -MF CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/mouse.cpp

CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/mouse.cpp > CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.i

CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/io/mouse.cpp -o CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.s

CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o: ../src/physics/rigidbody.cpp
CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o -MF CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/physics/rigidbody.cpp

CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/physics/rigidbody.cpp > CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.i

CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/physics/rigidbody.cpp -o CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.s

CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o: CMakeFiles/my_particle_system.dir/flags.make
CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o: ../src/physics/environment.cpp
CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o: CMakeFiles/my_particle_system.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o -MF CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o.d -o CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o -c /home/jaxe/Repositories/gpu_study/gpu_practice/src/physics/environment.cpp

CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_study/gpu_practice/src/physics/environment.cpp > CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.i

CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_study/gpu_practice/src/physics/environment.cpp -o CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.s

# Object files for target my_particle_system
my_particle_system_OBJECTS = \
"CMakeFiles/my_particle_system.dir/lib/stb.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/glad.c.o" \
"CMakeFiles/my_particle_system.dir/src/main.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/scene.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o" \
"CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o"

# External object files for target my_particle_system
my_particle_system_EXTERNAL_OBJECTS =

my_particle_system: CMakeFiles/my_particle_system.dir/lib/stb.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/glad.c.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/main.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/scene.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/algorithms/bounds.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/graphics/light.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/graphics/material.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/graphics/mesh.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/graphics/model.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/graphics/shader.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/graphics/texture.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/io/camera.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/io/joystick.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/io/keyboard.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/io/mouse.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/physics/rigidbody.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/src/physics/environment.cpp.o
my_particle_system: CMakeFiles/my_particle_system.dir/build.make
my_particle_system: lib/assimp-5.3.1/bin/libassimpd.so.5.3.0
my_particle_system: lib/assimp-5.3.1/contrib/zlib/libzlibstaticd.a
my_particle_system: CMakeFiles/my_particle_system.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Linking CXX executable my_particle_system"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_particle_system.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my_particle_system.dir/build: my_particle_system
.PHONY : CMakeFiles/my_particle_system.dir/build

CMakeFiles/my_particle_system.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/my_particle_system.dir/cmake_clean.cmake
.PHONY : CMakeFiles/my_particle_system.dir/clean

CMakeFiles/my_particle_system.dir/depend:
	cd /home/jaxe/Repositories/gpu_study/gpu_practice/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jaxe/Repositories/gpu_study/gpu_practice /home/jaxe/Repositories/gpu_study/gpu_practice /home/jaxe/Repositories/gpu_study/gpu_practice/build /home/jaxe/Repositories/gpu_study/gpu_practice/build /home/jaxe/Repositories/gpu_study/gpu_practice/build/CMakeFiles/my_particle_system.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/my_particle_system.dir/depend

