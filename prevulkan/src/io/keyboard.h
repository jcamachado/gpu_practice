#ifndef KEYBOARD_H
#define KEYBOARD_H

#include "../../lib/glad/glad.h" // include glad to get all the required OpenGL headers
#include <GLFW/glfw3.h>

class Keyboard{
    public:
        //key state callbacks        
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        
        // accessors

        static bool key(int key);
        static bool keyChanged(int key);
        static bool keyWentUp(int key);
        static bool keyWentDown(int key);
    private:
        static bool keys[];
        static bool keysChanged[];
};

#endif