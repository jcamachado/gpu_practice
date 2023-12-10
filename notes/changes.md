(This file has some code that were removed and may proof useful in the future)

/*  (comment on removed code On setting custom fbo in mainloop, because after this we will activate directional lights FBO)
            Render scene to the custom framebuffer
            Depth values will be output to the texture that we attach and 
            the color values will be aoutputted to the render buffer object
        */


(code removed for setting skybox and text on screen)
    Shader skyboxShader("assets/shaders/skybox/skybox.vs", "assets/shaders/skybox/sky.fs");
    // skyboxShader.activate();
    // skyboxShader.set3Float("min", 0.047f, 0.016f, 0.239f);
    // skyboxShader.set3Float("max", 0.945f, 1.000f, 0.682f);

    /*
        Skybox
    */
    Cubemap skybox;
    skybox.init();
    // skybox.loadTextures("assets/skybox");    // Load cubemap texture (image)
    ...
        (in the main loop)
        // skyboxShader.activate();
        // skyboxShader.setFloat("time", scene.variableLog["time"].val<float>());
        // skybox.render(skyboxShader, &scene); //Render skybox

        // scene.renderText(
        //     "comic", 
        //     textShader, 
        //     "Hello World!!", 
        //     50.0f, 
        //     50.0f, 
        //     glm::vec2(1.0f), 
        //     glm::vec3(0.5f, 0.6f, 1.0f)
        // );
        // scene.renderText(
        //     "comic", 
        //     textShader, 
        //     "Time: " + scene.variableLog["time"].dump(), 
        //     50.0f, 
        //     550.0f, 
        //     glm::vec2(1.0f), 
        //     glm::vec3(0.5f, 0.6f, 1.0f)
        // );
        // scene.renderText(
        //     "comic", 
        //     textShader, 
        //     "FPS: " + scene.variableLog["fps"].dump(), 
        //     50.0f, 
        //     550.0f - 40.0f, 
        //     glm::vec2(1.0f), 
        //     glm::vec3(0.5f, 0.6f, 1.0f)
        // );        
    (after mainloop)
    // skybox.cleanup();


(outline code removed from the main loop)

        if (scene.variableLog["displayOutline"].val<bool>()){
            /*
                glStencilMask tells opengl what to bitwise AND the stencil buffer with.
            */
            glStencilMask(0x00);                        // Disable stencil buffer writing for sphersd
            // scene.renderShader(outlineShader, false);    // Render outline
            // scene.renderInstances(sphere.id, outlineShader, dt);
        } 
        ...

        ...

        if (scene.variableLog["displayOutline"].val<bool>()){
            // Always write to stencil buffer with cubes
            glStencilFunc(GL_ALWAYS, 1, 0xFF);              // Set any stencil to 1
            glStencilMask(0xFF);                            // Always write to stencil buffer
            scene.renderInstances(cube.id, shader, dt);     // Render cubes

            glStencilFunc(GL_NOTEQUAL, 1, 0xFF); // render fragments if different than what is stored
            glStencilMask(0x00); // disable writing 
            glDisable(GL_DEPTH_TEST); // disable depth test so outlines are displayed behind objects

            scene.renderShader(outlineShader, false); // Render outline
            scene.renderInstances(cube.id, outlineShader, dt);

            // Reset valus
            glStencilFunc(GL_ALWAYS, 1, 0xFF);  // Every fragment written to stencil buffer
            glStencilMask(0xFF);                // Write always
            glEnable(GL_DEPTH_TEST);            // Re-enable depth test
        }
        else{
            // render cubes normally
            scene.renderInstances(cube.id, shader, dt);     // Render cubes
        }

(octree was not rendered [using box] anymore after new outline method)

(render lamps)
        scene.renderShader(lampShader, false);                  // Render lamps
        scene.renderInstances(lamp.id, lampShader, dt);
