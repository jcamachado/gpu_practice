Octree
Binary tree - Organiza o espaco de 1 dimensao para otimizacao de acesso a dados em relacoes de 1 pai com 2 filhos(2^1).
Quadtree - Organiza para um espaco de 2 dimensoes para otmizacao de acesso a dados em relacoes de 1 pai com 4 filhos (2^2).
Octree - Organizacao para um espaco de 3 dimensoes para otmizacao de acesso a dados em relacoes de 1 pai com 8 filhos (2^3).

Before the main loop:
-Construct (the root) node with initial objects 
-Build tree (sorting objects into its subdivisions)

During main loop (dynamic tree)
-Update object positions and properties
-Update tree with list of updated objects 

After loop
-Destroy tree

UML: 
Octree::node
----------------------
+parent: node
+objects: list
+children: node[]
+octants: unsigned char     //Usaremos States novamente para saber os quadrantes ativos para otimizar checagem de colisoes.
                            //E como unsigned char tem 8bits e cada no tem 8 filhos, da pra armazenar o estado de cada octante 
                            //Octante eh o analogo a quadrante para 3 dimensoes
+hasChildren: boolean
+treeReady: boolean
+treeBuild: boolean
+pending: queue             //For dynamic insertion (runtime)
+region: BoundingRegion
----------------------
+build(): void                          //before main loop
+update(): boid
+processPending(): void
+insert(obj: BoundingRegion): bool
+destroy(): void                        //after main loop