//DEFINITION OF MIXED PRECISION REFINEMENT
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
#include "cg_gpu_mp.h"


double mp_refinement(double *x, double *b, int maxiter, int iterMP, double relMP,int *status,int Nx,int Ny)
{
    int k;
    int nBytes = npts*sizeof(double);
    int nBytes_fl = npts*sizeof(float);
    double rn, rn0, relCG, iterCG;
    
    //HOST
    double *r,*z;
    float *r_fl,*z_fl;
    
    r=(double*)malloc(nBytes);
    z=(double*)malloc(nBytes);
    r_fl=(float*)malloc(nBytes_fl);
    z_fl=(float*)malloc(nBytes_fl);
      
    memset(x,0,nBytes);
    memset(r,0,nBytes);
    memset(z,0,nBytes);
    memset(r_fl,0,nBytes_fl);
    memset(z_fl,0,nBytes_fl);
    
    //DEVICE
    double *d_x,*d_b,*d_s,*d_r,*d_z,*d_xold;
     
    CHECK(cudaMalloc((void **)&d_x, nBytes));
    CHECK(cudaMalloc((void **)&d_b, nBytes)); 
    CHECK(cudaMalloc((void **)&d_s, nBytes));
    CHECK(cudaMalloc((void **)&d_r, nBytes));
    CHECK(cudaMalloc((void **)&d_z, nBytes));
    CHECK(cudaMalloc((void **)&d_xold, nBytes));
    
    CHECK(cudaMemset(d_x,0,nBytes));
    CHECK(cudaMemset(d_b,0,nBytes));
    CHECK(cudaMemset(d_s,0,nBytes));
    CHECK(cudaMemset(d_r,0,nBytes));
    CHECK(cudaMemset(d_z,0,nBytes));
    CHECK(cudaMemset(d_xold,0,nBytes));
    
    CHECK(cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice));

    //Inner tolerance setup
    if (relMP < 1e-06)
	  relCG=1e-04;
    else if((relMP<=1e-03) && (relMP>=1e-06))
	  relCG=1e-02;
    else if(relMP>1e-03)
	  relCG=1e-08;

    rn=norm_sqr(b);
    rn0=rn;
    status[0]=0;

    if (rn==0.0)
      return rn; 
    
    relMP*=relMP;
        
    k=1;
    while(k<iterMP)
    {    
       assign_v2v_gpu<<<grid,block>>>(d_xold,d_x,Nx,Ny);
        
       //r_k <- b-A_x_(k-1)  (DOUBLE PRECISION)
       laplace_2d_gpu<<<grid,block>>>(d_s,d_xold,Nx,Ny);
       add_gpu<<<grid,block>>>(d_r,d_b,-1.0,d_s,Nx,Ny);
      
       CHECK(cudaMemcpy(r, d_r, nBytes, cudaMemcpyDeviceToHost)); 
       rn=norm_sqr(r); 
       
        if ((rn/rn0)<=relMP)
        {
         break;
        } 
        
       //z_k <- A/r_k  (SINGLE PRECISION)
       d2fl(r_fl,r);
       iterCG=cg_gpu_mp(z_fl,r_fl,maxiter,relCG,status,Nx,Ny,nBytes_fl);

       //x_k <- x_(k-1)+z_k  (DOUBLE PRECISION)
       fl2d(z,z_fl);
       CHECK(cudaMemcpy(d_z, z, nBytes, cudaMemcpyHostToDevice));
       add_gpu<<<grid,block>>>(d_x,d_xold,1.0,d_z,Nx,Ny);
       
       printf("MP Iteration %d, CG Iterations %.0f, rel. norm. residuum: %e\n",k,iterCG,sqrt(rn/rn0));
      
	/*
       if ((rn/rn0)<=relMP)
        {
         break;
        }
	*/
       if(((relMP/(rn/rn0))>=1e-09) && (relMP<(rn/rn0)))
	   {
		relCG=relMP/(rn/rn0);
		printf("Iteration %d, new inner rel -> %e\n",k,sqrt(relCG));
	   }
       k++; 
      
    }

   printf("Rel. norm. residuum after %d MP iterations (%.0f CG Iterations) : %e\n",k,iterCG,sqrt(rn/rn0));

   if ((rn/rn0<=relMP) && (k<=maxiter))
      *status=k;
   if (rn/rn0>relMP)
      *status=-1;
   
    
   CHECK(cudaMemcpy(x, d_x, nBytes, cudaMemcpyDeviceToHost)); 
    
   //free(b);
   free(r);
   free(z);
   free(r_fl);
   free(z_fl);
    
   cudaFree(d_b);
   cudaFree(d_s);
   cudaFree(d_x);
   cudaFree(d_r);
   cudaFree(d_z);
   cudaFree(d_xold);
    
   return *x;
}