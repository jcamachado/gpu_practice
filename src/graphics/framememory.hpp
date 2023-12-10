#ifndef FRAMEMEMORY_HPP
#define FRAMEMEMORY_HPP

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>

#include "texture.h"

class FramebufferObject {
    public:
        /*
            -bitCombo: store states, for now: GL_COLOR_BUFFER_BIT and GL_DEPTH_BUFFER_BIT
        */
        GLuint val;
        GLuint width;
        GLuint height;
        GLbitfield bitCombo;

        std::vector<GLuint> rbos;
        std::vector<Texture> textures;
        
        FramebufferObject()
            : val(0), width(0), height(0), bitCombo(0) {}

        FramebufferObject(GLuint width, GLuint height, GLbitfield bitCombo)
            : val(0), width(width), height(height), bitCombo(bitCombo) {}

        void generate() {
            glGenFramebuffers(1, &val);
        }

        // This will allow us to bypass writing to the color buffer, which will save a lot of memory on the GPU
        void disableColorBuffer() {
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);      
        }

        void bind() {
            glBindFramebuffer(GL_FRAMEBUFFER, val);
        }

        void setViewport() {
            glViewport(0, 0, width, height);
        }

        void clear() {
            glClear(bitCombo);
        }

        void activate() {
            bind();
            setViewport();
            clear();
        }

        void allocateAndAttachRBO( GLenum attachType, GLenum format ) {
            GLuint rbo;

            // Generate
            glGenRenderbuffers(1, &rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, rbo);

            // Attach
            glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachType, GL_RENDERBUFFER, rbo);

            rbos.push_back(rbo);    // Add to list
        }

        void allocateAndAttachTexture(GLenum attachType, GLenum format, GLenum type) {
            std::string name = "tex" + textures.size(); // When deleting, always delete the last?
            Texture tex(name);
            
            // Allocate
            tex.bind();
            tex.allocate(format, width, height, type);
            //GL_CLAMP_TO_BORDER if we try to sample outside the texture, it will return the border color
            Texture::setParams(GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);

            // if its black, it will be considered on the shadow because a depth value of 0.0f is considered that the closest
            // object is right on the camera or on the light. And we just wan to set that thes no objects outside the shadow box
            // Nothing will be rendered in the shadow when were looking at the shadowbox
            float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f}; // White border, 
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

            // Attach
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachType, GL_TEXTURE_2D, tex.id, 0);

            textures.push_back(tex);
        }

        void attachTexture(GLenum attachType, Texture& tex) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachType, GL_TEXTURE_2D, tex.id, 0);
        }

        void cleanup() {
            // Delete RBOs
            glDeleteRenderbuffers(rbos.size(), &rbos[0]);    // Get pointer to first element because vector is contiguous

            // Delete generated textures
            for (Texture t : textures) {
                t.cleanup();
            }

            // Delete FBO
            glDeleteFramebuffers(1, &val);
        }

};


#endif