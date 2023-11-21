
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

NULL - Is a reserved key that basically translates to 0(zero).

Lambda function
    signature
        void traverse(void(*itemViewer)(T data)) {

    call:
    models.traverse([](Model* model) -> void {      // return is type(->) void
        model->cleanup();
    });

Classes
    constructor: To define class constructors, you gotta have a default construct defined. This could be an empty constructor: C(); or
        it have to all of its values set:  C(int a=1, int b=2);

    new: To pass the pointer of a new instance of an object, you have to say new C();
            ex: instances.push_back(new RigidBody(id, size, mass, pos));
        instances is a list of pointers. If it was a list of RigidBodys, this line would look like this:
            ex: instances.push_back(RigidBody(id, size, mass, pos));

Pointers
    nullptr - Object of "no value" for pointer