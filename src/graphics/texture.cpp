#include "texture.h"

#include <iostream>

Texture::Texture() {}

Texture::Texture(std::string name): name(name), type(aiTextureType_NONE) {
    generate(); // Dont load path, just generates slot on gpu for texture
}

Texture::Texture(std::string dir, std::string path, aiTextureType type)
    : dir(dir), path(path), type(type) {
        generate();
    }

void Texture::generate() {
    glGenTextures(1, &id);
}

void Texture::load(bool flip) {
    stbi_set_flip_vertically_on_load(flip);

    int width, height, nChannels;
    unsigned char* data = stbi_load((dir+"/"+path).c_str(), &width, &height, &nChannels, 0);

    GLenum colorMode = GL_RGB; // default
    switch(nChannels){
        case 1: colorMode = GL_RED; break;
        case 4: colorMode = GL_RGBA; break;
    }

    if (data) {
        glBindTexture(GL_TEXTURE_2D, id);
        glTexImage2D(GL_TEXTURE_2D, 0, colorMode, width, height, 0, colorMode, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    } else {
        std::cout << "Failed to load texture: " << path << std::endl;
    }

    stbi_image_free(data);
}

void Texture::bind() {
     // Bind texture to unit 0 by default. depends on the active texture unit
    glBindTexture(GL_TEXTURE_2D, id);
}
