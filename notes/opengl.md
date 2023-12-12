We can make changes to the shaderSrc before passing it to openGL.
We will add a header to the gls file.


### UBO (Uniform buffer object)

A new way of passing data through shaders
Here we have to have in mind that we are creating a new data structure to the system. We will allocate the data,
iterate over its members to translate values, and so on.

It relates to uniform data from shader files and buffer objects that pass data from the cpu to the gpu.
We can create "classes" of uniform objects.
It is useful for global variables between all shaders, like light values.
In light, the color values dont change so ofter, but with UBO we can insert what we want when things are updated. So for the positions we'll be able to update these certain values when they change.

Example of UBO in shader and the similarity of it with a cpp struct

-
layout (std140) uniform Test {
    int i;
    struct t2 {              // base alignment 16 (4N) matching the largest member
        int d[5];           // base alignment 16 (4N)
        float f;            // base alignment of N
        vec3 v[2];          // base alignment of 16 or 4N
    };
    struct t3 {     
        mat3 m;
        vec3 v;
    } [2];  // basically a matrix, 2,3 dimensions
};


To iterate over this, we will think of it as a tree data structure. where root then int is the first level left child
Depth first search. We will use a stack for this.
Each structure (vectors, arrays, structs) will be middle nodes of the tree. The leaves will be the scalars.
So, to iterate, we go through each element of the list, if it is a scalar, we found a leaf, if it is a struct, go further, left first.

In case of doubt, Michael Grieco explanation on video 44 middle of the video.. 
to iterate over struct
We have up to 4 levels, 0 to 3. (mywords) These levels are the depth of the increasing depth of but referencing the parent structure.
like in previous struct
struct{struct{d{5}}}. 2{1{0{0}}}. Depth index = 2 (3rd level) from root
In this scenario, inner struct is index 1 child of parent struct (int i is index 0), d is index 0 on inner struct and first value of d[5] is 
index 0 on its array.

keep track of the Depth ->                      stack[i].idx. Index at the ith element in the stack (ith pair), starting from depth 0
starting at the structure, depth = 0            stack[0] = 0
integer i depth = 0                             stack[0] = 0 zeroth lementh
struct t2, moves down 1 level, depth = 1,       stack[0] = 1 because we are on the first element ( In the new struct t2)
Down into int array d[5], depth = 2             stack[0] = 1 (struct), stack[1] = 0 (d 1st item in struct), stack[2] = 0 (0th element in array d)
for each element of d[5], still depth = 2 but   stack[0] = 1, stack[1] = 0, stack[2] = 0, then 1, 2, 3 and 4 for each element of d (5 elements)
Leave array d, decrease depth=1, go to float f  stack[0] = 1, stack[1] = 1 (2nd elemnt in struct)
vec3 v[2] is not primitive, increase depth = 2  stack[0] = 1, stack[1] = 2, stack[2] = 0 and 1
Leaves v2, decrease depth by 1, depth=1
no more elements in struct, decrease depth=0
next element from root(struct t3), in depth=0   stack[0] = 2,      
t3 is an array[2]. Enter array, depth=1
Enter t3 structure inside array, depth=2        stack[0] = 2, stack[1]=0 (first t3 in array)
enter matrix(index 0 in t3) mat3 m, depth = 3   stack[0] = 2, stack[1]=0, stack[2]=0 (m matrix), stack[3]=0,1,2 (matrix is an array of vectors, 3x3)
                                                                                                Why dont go further for each value? 
                                                                                                May he made a mistake?
                                                                                                like stack[3]=0,1,2 and stack[4]=0,1,2 since 3x3
                                                                                                **New understanding, mat3 is a mat[n] of vec3s
My understanding after this step is hazy, but is the end of video, dont know if he ommited some steps. But so far is understandable
goes back and to vec3 v, depth=2                stack[0] = 2, stack[1]=1, stack[2]=0, stack[3]=0,1,2 (vec3. Like this mat3 depth = vec3?)
Try to find another element, goes out of bounds and close the iterations.    

-
// all memory is aligned, but in layout std140 (probably the most popular standard), its not aligned the same way
struct Test {
    int i;
    struct t2{
        int d[5];
        float f;
        glm::vec3 v[2];
    };
    struct t3 {
        glm::mat3 m;
        glm::vec3 v;
    }
}



Base alignment and consumption (Pesquisar o Por que)


    N is the basic machine units a SCALAR consumes. The base alignment is N.
    if vector 2 or 4, consumes and the base alignment is 2N or 4N respectively.
    if vector 3, consumes 4N and the base alignment is 4N.

    Base alignment is the minimum number of bytes that can be allocated for the buffer object.
    So we can organize the elements and determine how to space it out in memory.

    if base alignment > offset, offset increased to base alignment multiple (of 16)
    struct will be rounded to a multiple of vec4 (4N -> 16)

    mat2x3  base alingn = 16 (basically is 2 vec 3, vec3 -> b.a. vec4 -> 16)

    always 16 at most
layout(std140) uniform Example {

                      // Base types below consume 4 basic machine units
                      //
                      //       base   base  align
                      // rule  align  off.  off.  bytes used
                      // ----  ------ ----  ----  -----------------------
        float a;      //  1       4     0    0    0..3
        vec2 b;       //  2       8     4    8    8..15
        vec3 c;       //  3      16    16   16    16..27
        struct {      //  9      16    28   32    (align begin)
          int d;      //  1       4    32   32    32..35
          bvec2 e;    //  2       8    36   40    40..47
        } f;          //  9      16    48   48    (pad end)
        float g;      //  1       4    48   48    48..51
        float h[2];   //  4      16    52   64    64..67 (h[0])
                      //                    80    80..83 (h[1])
                      //  4      16    84   96    (pad end of h)
        mat2x3 i;     // 5/4     16    96   96    96..107 (i, column 0)
                      //                   112    112..123 (i, column 1)
                      // 5/4     16   124  128    (pad end of i)
        struct {      //  10     16   128  128    (align begin)
          uvec3 j;    //  3      16   128  128    128..139 (o[0].j)
          vec2 k;     //  2       8   140  144    144..151 (o[0].k)
          float l[2]; //  4      16   152  160    160..163 (o[0].l[0])
                      //                   176    176..179 (o[0].l[1])
                      //  4      16   180  192    (pad end of o[0].l)
          vec2 m;     //  2       8   192  192    192..199 (o[0].m)
          mat3 n[2];  // 6/4     16   200  208    208..219 (o[0].n[0], column 0)
                      //                   224    224..235 (o[0].n[0], column 1)
                      //                   240    240..251 (o[0].n[0], column 2)
                      //                   256    256..267 (o[0].n[1], column 0)
                      //                   272    272..283 (o[0].n[1], column 1)
                      //                   288    288..299 (o[0].n[1], column 2)
                      // 6/4     16   300  304    (pad end of o[0].n)
                      //  9      16   304  304    (pad end of o[0])
                      //  3      16   304  304    304..315 (o[1].j)
                      //  2       8   316  320    320..327 (o[1].k)
                      //  4      16   328  336    336..347 (o[1].l[0])
                      //                   352    352..355 (o[1].l[1])
                      //  4      16   356  368    (pad end of o[1].l)
                      //  2       8   368  368    368..375 (o[1].m)
                      // 6/4     16   376  384    384..395 (o[1].n[0], column 0)
                      //                   400    400..411 (o[1].n[0], column 1)
                      //                   416    416..427 (o[1].n[0], column 2)
                      //                   432    432..443 (o[1].n[1], column 0)
                      //                   448    448..459 (o[1].n[1], column 1)
                      //                   464    464..475 (o[1].n[1], column 2)
                      // 6/4     16   476  480    (pad end of o[1].n)
                      //  9      16   480  480    (pad end of o[1])
        } o[2];
      };
mat3 -> 3 * vec3(12). Base align (rounded from 12 to 16).
https://registry.khronos.org/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt


UBO Example in main.cpp and instanced.vs

//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
layout (location = 0) in vec3 aPos; //the layout is the location of the vertex attribute in the VBO
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aOffset;
layout (location = 4) in vec3 aSize;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (std140) uniform Matrices {
    mat4 model2[3];
};

void main() {
    vec3 pos = aPos * aSize + aOffset;

    vs_out.FragPos = vec3(model * vec4(pos, 1.0)); // Its position in the world
    vs_out.Normal = mat3(transpose(inverse(model))) * aNormal; //Normal in world space

    gl_Position = projection * view * model2[2] * vec4(vs_out.FragPos, 1.0); //Order Matters!
    vs_out.TexCoord = aTexCoord;
}


    UBO::UBO ubo(0, {
        UBO::newColMatArray(3, 4, 4)
    });

    ubo.attachToShader(shader, "Colors");
    ubo.generate();
    ubo.bind();
    ubo.initNullData(GL_STATIC_DRAW);
    ubo.clear();

    ubo.bindRange();
    ubo.startWrite();

    Color colorArray[3] = {
        { {glm::vec3(1.0f, 0.0f, 0.0f)} },  // {struct{vec3}}
        { {glm::vec3(0.0f, 1.0f, 0.0f)} },
        { {glm::vec3(0.0f, 0.0f, 1.0f)} }
    };

    float fArr[3] = {
        0.0f, 0.0f, 0.0f
    };

    ubo.bind();
    // An array of matrices is an extended array of colums, so advancing throgh 8 colums model2[0] and model2[1] 
    // If each matrix was stored separately, then we would have to call advanceArray(4) 2 times.
    ubo.advanceArray(2*4);      
    glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, -5.0f));
    ubo.writeArrayContainer<glm::mat4, glm::vec4>(&m, 4);
    ubo.clear();