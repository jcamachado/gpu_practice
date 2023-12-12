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

#### PCF: Percentage closer filtering.
Is a technique to solve another problem regarding the finite nature of the texture sampled from the shadow mapping.

In this case, the problem happens not by rendering the regular directional light shadow on a surface, but the shadow
cast by other objects on the surface that is strongly aliased(pixelated). When you increases the shadow resolution, the pixelation decreases. 

To solve this we will have to do "blending".  That is a kind of average of values from neighboring coordinates on the texture. and average all 9 values.


### Shadow (Spotlight)

Will work similarly to the camera. Like the image cone of light has a canvas on the other side of objects.
Similar to what happens on update matrices in spotlight using perspective, but we will have 6 view matrices because the direction (in lookat) is what 
will change from face to face.

### Shadow (Point light)

Basically we will create 6 depthmaps using cubemaps.

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

### Geometry shader
    It comes between the vertex shader and the fragment shader. Its an optional step you don't need to actually compile anything with the geometry shader.
    But in geometry shader you can pass custom vertices based on the vertex shader into the fragment shader so everytime we call EmitVertex() it will take the value of glPosition and pass it into the fragment shader and it will set the color accordingly.

    In the example he calls EmitVertex twice (each after a gl_Position) and EndPrimitive() so openGL knows that we finished with the shape of the polygon, the line, the point, etc... and it could move on to the next primitive.

    We will use it on cubemap texture shadow for point lights.
    // Only geometry shader can emit many vertex from vertex shader

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



#### Normal mapping

Helps on creating quality texture
R G B are translated to X Y Z
So the blue value indicates the texture "depth", Z. And so on.

Rough explanation
But it has a limitation. The normal is pointing to the positive Z direction. So if the reflectiveness is in another axis, it will not display as well.
Possibly we would need to change the normal of the object or the Normal mapping so they could align.

### Tangent Spaces

Solution to the problem of our normal mapping only working on the Z direction (blue)
A coordinate space around our face per vertex, reducing the amount of calculation.
The tangent is calculated using the texture coordinates.
Each point   P has its <x, y, z> coordinates, but also have a <u,v> coordinates that are text coordinates
<x, y, z> vertex position
<u,v> = Texture Coordinates
We have '2 tangent' vectors: Tangente t, and Bitangent b;
T and B are IDEALLY perpendicular. Normal points out of the page.
Tangent is perpendicular to the face's normal  and going through the edges in the triangle. Which means pointing to the other 
2 points of the triangle.
We have to calculate tangent t and bitangent b.
Given a triangle (p1, p2, p3), lets consider p1's tangent.
p1: <x1, y1, z1> and <u1, u1>
p2: <x2, y2, z2> and <u2, u2>
Where the edges E1, E2 of p1:
E1 (p1, p2)
E2 (p1, p3)

//             deltaU  deltaV
dU1 = u2 - u1;  dV1 = v2 - v1
E1 = p2 - p1 = dU1 * t + dV1 * b =  
= <E1x, E1y, E1,> = <x2-x1, y2-y1, z2-z1> (E1 point values, world values for each vertex)


dU2 = u3 - u1;  dV = v3 - v1
E2 = p3 - p1 = dU2 * t + dV2 * b =
= <E2x, E2y, E2z>

We know deltaU, deltaV in edge coordinates because we can calculate from known values 
We gotta figure out t and b
_ t _, _ b _, ... are the vectors represented in the matrix
ps: i hope the dot and times products are correct. I based this on the 47 video's audio, since he
drew every product with a point(dot). and sometime [][] without symbol is a multiplication
matrix representation by me:
[a] are 1 matrix[3,1]
[b]
[c]

// Each edge is a linear combination of the tangent and the bitangent so, we can use this matrix form
where they are on top of each other
~~~
[_ E1 _]  = [dU1]*t + [dU1]*b   =
[_ E2 _]  = [dU2]     [dU2]

[E1x E1y E1z] = [dU1]*[tx ty tz] + [dU1]*[bx by bz]
[E2x E2y E2z]   [dU2]              [dU2]

E1x = [dU1 dV1] • [tx]
                  [bx]
E2x = [dU2 dV2] • [tx]
                  [bx]

Can be viewed as
[E1x] = [dU1 dV1] • [tx]        -> Matrix A • column cx
[E2x]   [dU2 dV2]   [bx]

[E1y] = A • cy
[E2y]   

[E1z] = A • cz
[E2z]   

Observation: we can represent:
    [.. .. .....]
A • [n1 n2 n3...] = [A•n1 A•n2 ...] 
    [.. .. .....]            

So,                 A
[E1x E1y E1z] = [dU1 dV1] * [tx ty tz]
[E2x E2y E2z]   [dU2 dV2]   [bx by bz]

(-1 is inverse)
A-1 * [_ E1 _] = A-1 * A -> goes out to identity matrix I
      [_ E2 _]

= I [_ t _] 
    [_ b _]

(detA os the determinant of A)
          -      A-1      -                
[_ t _] = (1/detA)[dU1 dV1][_ E1 _]          
[_ b _]           [dU2 dV2][_ E2 _]

~~~

### Interface blocks
    
The only purpose of this structure is to pass the vertex shader and the fragment shader or  the geometry shader. So its just to pass it between the shader files.
