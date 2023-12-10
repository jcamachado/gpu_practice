## Concepts
* VAO is the vertex array object, it stores the vertex attribute calls. VAO is how OpenGL call it. VAO dont make any changes to the vertex data. It only refers to the buffer object. Making it cheaper to access the data many times.
* VBO is the vertex buffer object, it stores the vertex data. VBO is how OpenGL call a Buffer Object.
* EBO is the Element buffer object It's a type of buffer object that you can store indices in, which you can then use to render primitives in OpenGL.

## Optics
The higher the field of view, the smaller objects will appear. 45o degrees FoV makes a cube looks larger then a 90o degrees FoV.

## Camera (movements)
row, yaw and pitch: Are the axis in what that should rotate certain object.  (plane reference)

Imagine an airplane rotating around 3 axes :
* pointing forward (roll)
* pointing east (pitch)
* pointing down (yaw) 

## Light 

#### Phong lighting

Os conceitos de aplicacao de diffuse light, directional light e specular light utilizados no codigo e' chamado de Phong Lighting.

    Diffuse light 
    // diffuse - is more about the position of the object in relation to the light source
    In point light (ambient light)
    // distance from the pointLight to the object to the object (fragment position) normal (incident ray)
    // intensity will be the result of the dot product of the normal and the light direction 
    In directional light
    //Different from pointLight, its not the position of light, but its direction
    //The dot product and light direction will be negative, since the light is coming from the opposite direction
    //And since negative dot product is discarded, we should multiply by -1

    Diffuse map: Ja que a cor da luz na superficie vem por uma multiplicacao da luz com a cor da superficie,
    o conceito de diffusemap abstrai nao apenas a cor natural do objeto, mas tambem engloba a textura do objeto.
    (Num primeiro momento do tutorial, ignoramos a cor do objeto se ele tiver uma textura, ja que ela provavelmente vai sobrepor a cor)


    Shininess: Multiplicada por um valor arbitrario em calculos (por vezes 128, 256). Pois seu valor de atribuicao
    varia de 0 a 1 (normalizado).

    Attenuation, a intensiodade da luz com relacao a distancia do ponto emissor de luz
    So consideremos para spotlight e pointlight. (Considerando directional light como a do sol, por exemplo
    a atenuacao eh desprezivel normalmente). Talveeeez pudesse ser considera em condicoes como 
    estacoes do ano. mas isso eh muito especifico e pouco recorrente.

    dist = lenght = (fragPos - lightPos)

    attenuation = 1 / (k0 + k1*d + k2*dist^2) 

    k0 = 1 for most purposes
    the bigger k1 and k2, the dimmer the light

* Problems with Phong: Specular is calculated by using dot product between reflection and view vectors. If the angle Θ (theta) between these vectors is greater than 90 degrees, the dot product will be negative.
And how we are dealing with it so far is that we ignore specular for degrees greater than 90.
So to compensate this, it got an upgraded version called Blinn-Phong lighting.

### Blinn-Phng lighting

Creates a halfway vector(**h**) between Viewer vector (**v**) and light source vector (**l**). The dot product between these vectors will always be < 90.

**h** = normalize(**l** + **v**)

### Shadows
#### Shadow Mapping (Directional Light)

Creates framebuffers on light source so, since it is a parallel source of light, we can say that if a position is occluse to the light's framebuffer, then the occluded object has a shadow cast into it. Another way to view the light framebuffer is like it had a camera, and it couldnt see some part of its quad. So this unseen part is in shadow.

In directional light, we will cast shadows that will look like a box around the caracter, since its parallel, and we dont need to cast where it wont be rendered, only within the player(camera) field of view. We will call this box: Bounding region br. ()in light.h

#### Shadow acne
Happens when we have finite resolution on texture

When a light maps to a surface by its depth and with finite resolution texture, we consider a hit on one pixel.
Every fragment can be so sharf of a resolution, and this process keeps going to for "parallel light vectors"
When we take the coordinate from each fragment to determine if its in the shadow or not, we convert it into an integer (into a normalized coordinate), and each normalized coordinate will map to some integer pixel coordinate on the image.

Minhas palavras / a partida da explicacao visual do MGrieco
A depthmap on a surface is kinda perpendicular to light vector. So, picturing this as a diagonal relatively to the surface.  
And imagine that this diagonal starts under the surface and goes further in depth above the surface. 
The first part will be counted as out of the shadow, and the other part as being shadowed.  This makes a striped pattern.
But i'm still a bit confused.

This will be solved by a solution called Bias where we just offset it.

(from stackoverflow)
Shadow acne is caused by the discrete nature of the shadow map. A shadow map is composed of samples, a surface is continuous. Thus, there can be a spot on the surface where the discrete surface is further than the sample. The problem does persist even if you multi sample, but you can sample smarter in ways that can nearly eliminate this at significant cost.

The canonical way to solve this is to offset the shadow map slightly so the object no longer self shadows itself. This offset is called a bias. One can use more smart offsets than just a fixed value but a fixed value works quite well and has minimal overhead.

### Gamma Correction

We expect that the color input to the system to be output on the screen. But this is not what happens.
The function of outputColor(inputColor) will be referred as **"Expected color" E(x)**.
We expect that this function E(x) to be linear. But in monitors, this functions are usually exponential, quadratic.
**Monitor output -> M(x)**. Where this function usually is: M(x) = x<sup>γ</sup> (γ is gamma)

So, to make things closer to the ideal and nullify this effect, we have to create our own **Gamma correction function C(x)**, where
C(x) = x<sup>1/γ</sup> 

Example:

* E(0.5) = 0.5
* M(0.5) = 0.5<sup>γ</sup>
* C(0.5) = 0.5<sup>1/γ</sup> 

And we pass C into M function, so:

* (0.5<sup>1/γ</sup>)<sup>γ</sup> = 0.5

## Models
    Models are divided into units called MESHES (A section of the model)
    So in a body, the Meshes could be the head, limbs and so on. 

## Shaders
    Aparentemente qualquer shader pode fazer qualquer coisa. Essa diferenciacao parece mais organizacional (usando opengl) 

    -Fragment shader(fs): basicamente o que eh pintado sobre os pixels. No caso, de triangulos.
    Entao as funcoes que se relacionam com elementos da tela alterarao a cor dos pixels desses elementos
    -Vertex shader(vs): Manipula posicao de vertices. 


# OPENGL
File extensions
-.fs fragment shader 
-.vs vector shader
*Both use a simplified version of C

lightDir is 

    float diff = max(dot(norm, lightDir), 0.0); //when light is perpendicular to the normal, dot product is 0


Takes the normal vector and updates its position based on the possibly updated model position
Normal = mat3(transpose(inverse(model))) * aNormal; //Normal in world space

Cubemaps: More convenient to use since it only takes one textuer slot. We add 6 textures, tie them together so they only take 1 texture slot.
And since cube and cubemap have the same vertices, we only have to pass one set of vertices to be drawn by the gpu.

Stencil: A mapping of a renderable view(matrix) and a stencil matrix of 1 and 0s. If the values on the second matrix is 0, the corresponding value in the original  view matrix wont be drawn, otherwise, draw normally.

Framebuffer: Combines the color, depth and stencil buffers and it displays it somewhere. The default framebuffer is on screen and glfw sets it up for us. And connects this framebuffer to the monitor.