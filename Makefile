include .env

# Needs to add tinyobjloader to the project, here it is linked through the usr/include directory
CFLAGS = -std=c++17 -O2		# 02 is optimization level, but for development, remove it to run faster 
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

# Will cause "no such file" warning when no changes are made to the file
vertSources = $(shell find ./src/shaders -type f -name "*.vert") 
# vertSources = $(shell find ./shaders -type f -name "*.vert") 
vertObjFiles = $(patsubst %.vert, %.vert.spv, $(vertSources))
fragSources = $(shell find ./src/shaders -type f -name "*.frag")
# fragSources = $(shell find ./shaders -type f -name "*.frag")
fragObjFiles = $(patsubst %.frag, %.frag.spv, $(fragSources))


TARGET = ./build/gpu_practice
$(TARGET): $(vertObjFiles) $(fragObjFiles) src/*.cpp
	g++ $(CFLAGS) -o $(TARGET) $^ $(LDFLAGS)

# make shader targets
%.spv: %
	$(GLSLC) $< -o $@

build: src/*.cpp
	echo "Building..."
	g++ $(CFLAGS) -o build/gpu_practice $^ $(LDFLAGS)
	echo "Done."

.PHONY: run clean build run full-run

run: ./build/gpu_practice
	./build/gpu_practice

clean:
	rm -f build/gpu_practice

full-run: clean build run
	

.DEFAULT_GOAL := $(TARGET)


# build: src/*.cpp
# 	echo "Building..."
# 	g++ $(CFLAGS) -o build/gpu_practice $^ $(LDFLAGS)
# 	echo "Done."

# .PHONY: run clean build

# run: ./build/gpu_practice
# 	./build/gpu_practice

# clean:
# 	rm -f build/gpu_practice
