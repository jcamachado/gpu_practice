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


Para executar o codigo, basta executar main.cpp, e ter instalado na maquina opengl 3.3 e seus frameworks apontados no CMakeLists.txt and Assimp(Que deve dar mais dor de cabeca).
Abaixo tem alguns detalhes. Estou a disposicao.




===== 

OpenGL
Disclaimer:
This project's OpenGL base is heavily based and inspired by the works of Michael Grieco [https://github.com/michaelg29/yt-tutorials/tree/master/CPP/OpenGL], [https://michaelg29.github.io/] and of Joey de Vries (learnopengl.com).
Including some algorithms and files being directly copied from Michael's repository, such as avl, trie, cmathematics, jsoncpp and others. (Thank you for the tutorial!!!)

Assimp is not uploaded (just indication where to place its files)
Follow this to get to the correct path (also, the assimp-M-m-n folder is the original downloaded one, the other was composed just like on the video below suggests from building) 
https://www.youtube.com/watch?v=oci7xJEg6sU&list=PLysLvOneEETPlOI_PI4mJnocqIpr2cSHS&index=19

Models so far:
1 - https://sketchfab.com/3d-models/lotr-troll-for-animatingrigging-f4d777fd41d045fb8692f19f07b998fe
2 - https://github.com/michaelg29/yt-tutorials/tree/master/CPP/OpenGL/OpenGLTutorial/OpenGLTutorial/assets/models/m4a1
3 - Sphere: https://github.com/michaelg29/yt-tutorials/tree/master/CPP/OpenGL/OpenGLTutorial/OpenGLTutorial/assets/models/sphere


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

