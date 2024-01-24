CFLAGS = -std=c++17 -O2		# 02 is optimization level, but for development, remove it to run faster 
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

VulkanTest: src/*.cpp
	g++ $(CFLAGS) -o build/gpu_practice $^ $(LDFLAGS)

.PHONY: test clean

test: ./build/gpu_practice
	./build/gpu_practice

clean:
	rm -f build/gpu_practice
