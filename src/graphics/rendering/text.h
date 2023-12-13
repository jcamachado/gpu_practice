#ifndef TEXT_H
#define TEXT_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <glm/glm.hpp>

#include <string>
#include <map>

#include "../memory/vertexmemory.hpp"
#include "../rendering/shader.h"

/*
    A character is created using a set of metrics.
    Distance is measured in pixels. Hence the int values.
    -Origin: Is the botton left corner of the object (its not the lower most point of the object)
    -Advance: Is the distance from origin of current glyph to the origin of next glyph (space between characters)
    We will be using only the x component of the advance. The y component is always 0.
    The advance is measured in 1/64 pixels. 
    -Bearing (x, y): Is the distance from the origin to the top left corner of the glyph
    -Height: Is the height of the glyph. Distance between the topmost pixel and the bottommost pixel (ymax - ymin)
    -Width: Is the width of the glyph. Distance between the leftmost pixel and the rightmost pixel (xmax - xmin)
*/
typedef struct {
    unsigned int textureId; // texture id storing character
    glm::ivec2 size;        // size of glyph
    glm::ivec2 bearing;     // offset from baseline to left/top of glyph
    unsigned int advance;   // horizontal offset to advance to next glyph

} Character;

class TextRenderer{
    public:
        TextRenderer();
        TextRenderer(int height);

        bool loadFont(FT_Library& ft, std::string path);

        void render(Shader shader, std::string text, float x, float y, glm::vec2 scale, glm::vec3 color);

        void cleanup();

    private:
        int height;

        std::map<char, Character> chars;

        /*
            Our goal is to map a 2d quad to a 3d space
            So we need a VAO to store positions and texture coordinates. Like a model.
            We will use dynamic draw. 
        */
        ArrayObject VAO;        
};

#endif