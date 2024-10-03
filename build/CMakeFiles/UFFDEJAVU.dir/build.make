# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_SOURCE_DIR = /home/jaxe/Repositories/gpu_practice

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jaxe/Repositories/gpu_practice/build

# Include any dependencies generated for this target.
include CMakeFiles/UFFDEJAVU.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/UFFDEJAVU.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/UFFDEJAVU.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/UFFDEJAVU.dir/flags.make

CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o: /home/jaxe/Repositories/gpu_practice/src/buffer.cpp
CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/buffer.cpp

CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/buffer.cpp > CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/buffer.cpp -o CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o: /home/jaxe/Repositories/gpu_practice/src/camera.cpp
CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/camera.cpp

CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/camera.cpp > CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/camera.cpp -o CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o: /home/jaxe/Repositories/gpu_practice/src/descriptors.cpp
CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/descriptors.cpp

CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/descriptors.cpp > CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/descriptors.cpp -o CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o: /home/jaxe/Repositories/gpu_practice/src/device.cpp
CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/device.cpp

CMakeFiles/UFFDEJAVU.dir/src/device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/device.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/device.cpp > CMakeFiles/UFFDEJAVU.dir/src/device.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/device.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/device.cpp -o CMakeFiles/UFFDEJAVU.dir/src/device.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o: /home/jaxe/Repositories/gpu_practice/src/first_app.cpp
CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/first_app.cpp

CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/first_app.cpp > CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/first_app.cpp -o CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o: /home/jaxe/Repositories/gpu_practice/src/game_object.cpp
CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/game_object.cpp

CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/game_object.cpp > CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/game_object.cpp -o CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o: /home/jaxe/Repositories/gpu_practice/src/keyboard_movement_controller.cpp
CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/keyboard_movement_controller.cpp

CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/keyboard_movement_controller.cpp > CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/keyboard_movement_controller.cpp -o CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o: /home/jaxe/Repositories/gpu_practice/src/main.cpp
CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/main.cpp

CMakeFiles/UFFDEJAVU.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/main.cpp > CMakeFiles/UFFDEJAVU.dir/src/main.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/main.cpp -o CMakeFiles/UFFDEJAVU.dir/src/main.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o: /home/jaxe/Repositories/gpu_practice/src/model.cpp
CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/model.cpp

CMakeFiles/UFFDEJAVU.dir/src/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/model.cpp > CMakeFiles/UFFDEJAVU.dir/src/model.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/model.cpp -o CMakeFiles/UFFDEJAVU.dir/src/model.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o: /home/jaxe/Repositories/gpu_practice/src/pipeline.cpp
CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/pipeline.cpp

CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/pipeline.cpp > CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/pipeline.cpp -o CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o: /home/jaxe/Repositories/gpu_practice/src/renderer.cpp
CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/renderer.cpp

CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/renderer.cpp > CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/renderer.cpp -o CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o: /home/jaxe/Repositories/gpu_practice/src/swap_chain.cpp
CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/swap_chain.cpp

CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/swap_chain.cpp > CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/swap_chain.cpp -o CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o: /home/jaxe/Repositories/gpu_practice/src/systems/multiview_render_system.cpp
CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/systems/multiview_render_system.cpp

CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/systems/multiview_render_system.cpp > CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/systems/multiview_render_system.cpp -o CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o: /home/jaxe/Repositories/gpu_practice/src/systems/point_light_system.cpp
CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/systems/point_light_system.cpp

CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/systems/point_light_system.cpp > CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/systems/point_light_system.cpp -o CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o: /home/jaxe/Repositories/gpu_practice/src/systems/simple_render_system.cpp
CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/systems/simple_render_system.cpp

CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/systems/simple_render_system.cpp > CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/systems/simple_render_system.cpp -o CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.s

CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o: CMakeFiles/UFFDEJAVU.dir/flags.make
CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o: /home/jaxe/Repositories/gpu_practice/src/window.cpp
CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o: CMakeFiles/UFFDEJAVU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o -MF CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o.d -o CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o -c /home/jaxe/Repositories/gpu_practice/src/window.cpp

CMakeFiles/UFFDEJAVU.dir/src/window.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/UFFDEJAVU.dir/src/window.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaxe/Repositories/gpu_practice/src/window.cpp > CMakeFiles/UFFDEJAVU.dir/src/window.cpp.i

CMakeFiles/UFFDEJAVU.dir/src/window.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/UFFDEJAVU.dir/src/window.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaxe/Repositories/gpu_practice/src/window.cpp -o CMakeFiles/UFFDEJAVU.dir/src/window.cpp.s

# Object files for target UFFDEJAVU
UFFDEJAVU_OBJECTS = \
"CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o" \
"CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o"

# External object files for target UFFDEJAVU
UFFDEJAVU_EXTERNAL_OBJECTS =

UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/buffer.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/camera.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/descriptors.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/device.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/first_app.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/game_object.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/keyboard_movement_controller.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/main.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/model.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/pipeline.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/renderer.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/swap_chain.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/systems/multiview_render_system.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/systems/point_light_system.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/systems/simple_render_system.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/src/window.cpp.o
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/build.make
UFFDEJAVU: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
UFFDEJAVU: /usr/lib/x86_64-linux-gnu/libvulkan.so
UFFDEJAVU: CMakeFiles/UFFDEJAVU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jaxe/Repositories/gpu_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Linking CXX executable UFFDEJAVU"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/UFFDEJAVU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/UFFDEJAVU.dir/build: UFFDEJAVU
.PHONY : CMakeFiles/UFFDEJAVU.dir/build

CMakeFiles/UFFDEJAVU.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/UFFDEJAVU.dir/cmake_clean.cmake
.PHONY : CMakeFiles/UFFDEJAVU.dir/clean

CMakeFiles/UFFDEJAVU.dir/depend:
	cd /home/jaxe/Repositories/gpu_practice/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jaxe/Repositories/gpu_practice /home/jaxe/Repositories/gpu_practice /home/jaxe/Repositories/gpu_practice/build /home/jaxe/Repositories/gpu_practice/build /home/jaxe/Repositories/gpu_practice/build/CMakeFiles/UFFDEJAVU.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/UFFDEJAVU.dir/depend

