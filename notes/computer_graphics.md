* Concepts *
// VAO is the vertex array object, it stores the vertex attribute calls
// VBO is the vertex buffer object, it stores the vertex data
// EBO is the Element buffer object It's a type of buffer object that you can store indices in, which you can then use to render primitives in OpenGL.

Optics
The higher the field of view, the smaller objects will appear. 45o degrees FoV makes a cube looks larger then a 90o degrees FoV.

Camera (movements)
row, yaw and pitch: Are the axis in what that should rotate certain object.  (plane reference)
Imagine an airplane rotatin around 3 axes :
pointing forward (roll)
pointing east (pitch)
pointing down (yaw) 

Light
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

Models
    Models are divided into units called MESHES (A section of the model)
    So in a body, the Meshes could be the head, limbs and so on. 

Shaders
    Aparentemente qualquer shader pode fazer qualquer coisa. Essa diferenciacao parece mais organizacional (usando opengl) 

    -Fragment shader(fs): basicamente o que eh pintado sobre os pixels. No caso, de triangulos.
    Entao as funcoes que se relacionam com elementos da tela alterarao a cor dos pixels desses elementos
    -Vertex shader(vs): Manipula posicao de vertices. 


*  OPENGL *
File extensions
-.fs fragment shader 
-.vs vector shader
*Both use a simplified version of C

lightDir is 
float diff = max(dot(norm, lightDir), 0.0); //when light is perpendicular to the normal, dot product is 0


Takes the normal vector and updates its position based on the possibly updated model position
Normal = mat3(transpose(inverse(model))) * aNormal; //Normal in world space