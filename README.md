# CP3_CGProyect

In the present work, the popular iterative conjugate gradient method was adapted to parallel programing using CUDA, introducing a few ways to enhance the normal algorithm by means of preconditioning and mixed precision refinement iteration. 

The power iteration was applied to determine the maximum and minimum eigenvalues of a system of linear equations, in particular, to the 2D discretization of the Laplace operator.

The code consists of several files that need to be compiled and linked in a runnable file. 
This can be achieved using the `cmake` tool with the `CMakeLists.txt` file.

## Compile

To run the code, first a `Makefile` is produced from the `CMakeLists.txt` file with the following command:

```
cmake .
```

This line only needs to be run at the beginning of the execution, unless the `CMakeLists.txt` is modified.
The command `make` then uses it to compile and link all the other files. This generates the runnable file `run-cg`, which is used to run the programm:
```
make
./run-cg
```


## Adding files

To modify, fix, or add functions to the programm, the code to the functions has to be added in the `CMakeLists.txt` file:
```
...
      cg.cu
      cg.h
      new.cu
      new.h
   )
...
```

