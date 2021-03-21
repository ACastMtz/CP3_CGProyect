/*********************************************************************
Computes the scalar product of two vectors using a CPU function and a GPU kernel, to check the implementation of dot_prod_kernel
**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <ctime>
#include "global.h"
#include "geometry.h"
#include "linalg.h"
#include "common.h"
#include "transform.h"


void dot_prod_test(double *a, double *b, int nBytes)
{
   double *h_res;
   h_res=(double*)malloc(sizeof(double));
   
   // Device allocation
   double *d_a, *d_b, *d_res, iStart, iElaps;
    
   CHECK(cudaMalloc((void **)&d_a, nBytes));
   CHECK(cudaMalloc((void **)&d_b, nBytes));
   CHECK(cudaMalloc((void **)&d_res, sizeof(double)));

   CHECK(cudaMemset(d_a,0,nBytes));
   CHECK(cudaMemset(d_b,0,nBytes));
   CHECK(cudaMemset(d_res,0,sizeof(double)));
  
   // transfer data from host to device
   CHECK(cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice));

  //Execution configuration for dot-product-kernel  
   int Nunroll2=8;
   dim3 block2 (256,1);  
   int nblk = (npts + (block2.x*Nunroll2) - 1)/(block2.x*Nunroll2);
   dim3 grid2 (nblk,1);   


   //CPU version
    double result = 0.0;
    iStart = seconds();
    result=vector_prod(a,b);
    iElaps = seconds() - iStart;
    printf("\n dot product computed on CPU is: %.8f and took: %f s", result, iElaps);

   //GPU version
	iStart = seconds();
dot_product_kernel<<<grid2,block2>>>(d_a,d_b,d_res, npts);
CHECK(cudaMemcpy(h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost));
iElaps = seconds() - iStart;
printf("\ndot product computed on GPU is: %.8f and took: %f s", *h_res, iElaps); 

    
    
    //Free meory
    free(h_res);
    cudaFree(d_res);
    cudaFree(d_a);
    cudaFree(d_b);
}