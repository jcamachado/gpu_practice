# Define the build directory
BUILD_DIR = build

# Define the target executable name
TARGET = UFFDEJAVU

# Define the source directory (where your CMakeLists.txt is located)
SRC_DIR = $(shell realpath .)

# Define the shader directory and output directory
SHADER_DIR = shaders
SHADER_BUILD_DIR = $(BUILD_DIR)/shaders

# Default target: configure, build, compile shaders, and run
all: configure build compile_shaders run

# Configure the project using cmake
configure:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(SRC_DIR)

# Build the project using make
build: configure
	cd $(BUILD_DIR) && make
	
# Compile shaders
compile_shaders:
	mkdir -p $(SHADER_BUILD_DIR)
	for shader in $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag; do \
		if [ -f $$shader ]; then \
			glslc $$shader -o $(SHADER_BUILD_DIR)/$$(basename $$shader .vert).spv; \
			glslc $$shader -o $(SHADER_BUILD_DIR)/$$(basename $$shader .frag).spv; \
			glslc $$shader -o $(SHADER_BUILD_DIR)/$$(basename $$shader .geom).spv; \
		fi \
	done

# Run the resulting executable
run: build
	./$(BUILD_DIR)/$(TARGET)

# Clean the build directory
clean:
	rm -rf $(BUILD_DIR)