CFLAGS = -std=c++17 -O2		# 02 is optimization level, but for development, remove it to run faster 
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vbuild: src/*.cpp
	echo "Building..."
	g++ $(CFLAGS) -o build/gpu_practice $^ $(LDFLAGS)
	echo "Done."

.PHONY: test clean

run: ./build/gpu_practice
	./build/gpu_practice

clean:
	rm -f build/gpu_practice
