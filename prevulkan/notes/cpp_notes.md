
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

void foo(CDummy& x);
//you pass x by reference
//if you modify x inside the function, the change will be applied to the original variable
//a copy is not created for x, the original one is used
//this is preffered for passing large objects
//to prevent changes, pass by const reference:
void fooconst(const CDummy& x);

Should you write it like this: void myfunc(int *a) or like this void myfunc(int &a)?


Pointers (ie. the '*') should be used where the passing "NULL" is meaningful. For example, you might use a NULL to represent that a particular object needs to be created, or that a particular action doesn't need to be taken. Or if it ever needs to be called from non-C++ code. (eg. for use in shared libraries)

eg. The libc function time_t time (time_t *result);

If result is not NULL, the current time will be stored. But if result is NULL, then no action is taken.

If the function that you're writing doesn't need to use NULL as a meaningful value then using references (ie. the '&') will probably be less confusing - assuming that is the convention that your project uses.

\* = pointer
& = reference