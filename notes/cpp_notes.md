
File extension 
-.h: header        
-  - C/C++ compatible or pure C Headers
-  -  - This header can be included by both a C source, and a C++ source, directly or indirectly.
-.hpp: C++ Headers
- - Also a header but for c++ only. Includes definitions for classes, functions, constants and such declarations.
-.cpp: c++ source file.
-.c: c source file.
*looks like the MC is using hpp in this project kinda wrong differently from what people explain on internet. He uses liek a header+source file.

Iteration
-for (tipo unidade: lista), parecido com Python-
    for(Mesh mesh : meshes){
        meshes.render(shader);
    }

Lambda function
-signature
void traverse(void(*itemViewer)(T data)) {

-call:
models.traverse([](Model* model) -> void {      // return is type(->) void
        model->cleanup();
    });