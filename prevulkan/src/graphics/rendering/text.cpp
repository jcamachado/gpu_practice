#include "text.h"

TextRenderer::TextRenderer(){}

TextRenderer::TextRenderer(int height) :
    height(height){}

bool TextRenderer::loadFont(FT_Library& ft, std::string path){
    FT_Face fontFace;
    if (FT_New_Face(ft, path.c_str(), 0, &fontFace)){
        //FT_New_Face return 0 if successful
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return false;
    }
    
    /*
        Set height and width.
        -The width is set to 0 to calculate the width based on the height dynamically
    */
    FT_Set_Pixel_Sizes(fontFace, 0, height);

    /*
        Load texture
        -Disabling byte-alignment restriction so that we can use 1 byte per pixel
    */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Disable byte-alignment restriction

    for (unsigned char c = 0; c < 128; c++){
        // Load character glyph 
        if (FT_Load_Char(fontFace, c, FT_LOAD_RENDER)){
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            continue;
        }

        /*
            Generate texture
            -Different from before, the texture does not come from an image file. Chacters textures
            comes from the fontFace->glyph->bitmap.buffer
        */
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(
            GL_TEXTURE_2D,                  // Mode, target
            0,                              // Mipmap level
            GL_RED,                         // 1 byte per pixel
            fontFace->glyph->bitmap.width,  // Width
            fontFace->glyph->bitmap.rows,   // Height
            0,                              // Border
            GL_RED,                         // Format
            GL_UNSIGNED_BYTE,               // Type        
            fontFace->glyph->bitmap.buffer  // Data (texture class)
        );

        // Texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);       // Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Now store character for later use
        chars[c] = {
            texture,
            glm::ivec2(fontFace->glyph->bitmap.width, fontFace->glyph->bitmap.rows),
            glm::ivec2(fontFace->glyph->bitmap_left, fontFace->glyph->bitmap_top),
            (unsigned int)fontFace->glyph->advance.x
        };
    }
    FT_Done_Face(fontFace);

    // Setup VAO
    VAO.generate();
    VAO.bind();
    
    // Setup VBO
    VAO["VBO"] = BufferObject(GL_ARRAY_BUFFER);
    VAO["VBO"].generate();
    VAO["VBO"].bind();

    // To set the texture, we need to pass a position and a texture coordinate (5 floats)
    // And glm doesnt have a vec5, so we need to use vec4
    // 2D quad requires 6 vertices of 4 floats each (2 for pos, 2 for tex coords)
    VAO["VBO"].setData<GLfloat>(6 * 4, NULL, GL_DYNAMIC_DRAW);
    VAO["VBO"].setAttrPointer<GLfloat>(0, 4, GL_FLOAT, 4, 0);

    VAO["VBO"].clear();

    ArrayObject::clear();
    return true;
}

// Text will be rendered using orthographic projection
void TextRenderer::render(
    Shader shader,
    std::string text, 
    float x, 
    float y, 
    glm::vec2 scale, 
    glm::vec3 color)
{
    shader.activate();
    // GLint textColorLocation = glGetUniformLocation(shaderProgram, "textColor");
    // glUniform3f(textColorLocation, color.x, color.y, color.z);
    shader.set3Float("textColor", color.x, color.y, color.z);

    glActiveTexture(GL_TEXTURE0);

    
    VAO.bind();

    for (int i = 0, len = text.size(); i < len; i++){
        Character c = chars[text[i]];
        
        float xPos = x + c.bearing.x * scale.x;
        float yPos = y - (c.size.y - c.bearing.y) * scale.y;

        float width = c.size.x * scale.x;
        float height = c.size.y * scale.y;

        float vertices[6 * 4] = {
            xPos,           yPos + height,  0.0f, 0.0f,
            xPos,           yPos,           0.0f, 1.0f,
            xPos + width,   yPos,           1.0f, 1.0f,

            xPos,           yPos + height,  0.0f, 0.0f,
            xPos + width,   yPos,           1.0f, 1.0f,
            xPos + width,   yPos + height,  1.0f, 0.0f
        };

        glBindTexture(GL_TEXTURE_2D, c.textureId);

        glBindBuffer(GL_ARRAY_BUFFER, 1);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        x += (c.advance >> 6) * scale.x;
    }

    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
}
void TextRenderer::cleanup(){
    VAO.cleanup();
}