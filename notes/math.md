### Vectors
cross function, to get the ortogonal vector between 2 vectors

# Matrices
    Matrix multiplication order for objects onto the screen
    //position
    //obj coords           placed in context           camera                what we see
    //local   -----model----> world ------------view--> view ---projection--> clip   
    //Multiplication must be transformation * model * position
    //projection * view * model * position 
    //transformation multiplication order is translate, rotate and scale
    Procurar sobre as matrizes de transformacao

    To pass a matrix from onde coordinate to another, you need to Transpose the Inverted original Matrix.
    So, get model M, invert it to M-1 and then transpose it.
    

