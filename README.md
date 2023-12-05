# gpu_practice

Environment:
Ubuntu 22.04
g++ 11.4.0
gcc 11.4.0 x64
gnu 17
gnu++ 14
nvcc 11.5.119
NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2

Aluno: Jose Carlos de Almeida Machado


Hello, aqui vou disponibilizar codigos feitos para treinar CUDA e prog. paralela.


Aula 1 e 2: ./openmp - 1/ e ./openacc - 2

Aula 3: Esqueleto - ./cuda/particleSystem/*

Aula 4 - Soma de matrizes: ./cuda/matrixSum, processamento de imagem (tornar Grayscale) ./cuda/imageShorteningParallel.cu

Aula 5 - Shared Memory - ./cuda/exercicios de aula/sharedMem1.cu 

Aula 6 - Reduce with Shared Memory - ./cuda/exercicios de aula/reduce.cu 

Aula 8 - Sent separately



===== 

OpenGL

Assimp is not uploaded (just indication where to place its files)
Follow this to get to the correct path (also, the assimp-M-m-n folder is the original downloaded one, the other was composed just like on the video below suggests from building) 
https://www.youtube.com/watch?v=oci7xJEg6sU&list=PLysLvOneEETPlOI_PI4mJnocqIpr2cSHS&index=19

Models so far:
1 - https://sketchfab.com/3d-models/lotr-troll-for-animatingrigging-f4d777fd41d045fb8692f19f07b998fe
2 - https://sketchfab.com/3d-models/low-poly-m4a1-8cab1cbeb82c4396a154f9fc8771417b
3 - https://sketchfab.com/3d-models/sphere-b31b12ffa93a40f48c9d991b6f168f4d



Transitioning the compiler to clang 
-Last commit using gcc: commit ff4608ba74840e8611a03dfdf5b3ba66bee3578c
-Delete all previous build/ content
sudo apt install libstdc++-12-dev
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

Installed libs (some others may be necessary to install):
sudo apt-get install libgl1-mesa-dev (for opengl)
sudo apt-get install libharfbuzz-dev (for freetype)

Manually included libs
GLAD : glad.c on same level as main.cpp. Also, might change the reference of glad.h inside glad.c dependending on their relative directories.
ASSIMP (included manually): Gotta copy the assimp-x-y-z/build/include to an /lib/assimp/ folder. assimp/ and assimp-x-y-z/ must be in the same directory.
    Where x, y and z are Major, minor and patch versions.
FREETYPE (included manually)
stb (included manually)

