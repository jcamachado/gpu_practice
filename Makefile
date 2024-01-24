CFLAGS = -std=c++17 -O2		# 02 is optimization level, but for development, remove it to run faster 
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

VulkanTest: main.cpp
	g++ $(CFLAGS) -o gpu_practice main.cpp $(LDFLAGS)

.PHONY: test clean

test: gpu_practice
	./gpu_practice

clean:
	rm -f VulkanTest
