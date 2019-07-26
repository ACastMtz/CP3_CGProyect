//DEFINITION OF MIXED PRECISION REFINEMENT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"


double mp_refinement(float *x_fl, float *b_fl, int maxiter,double rel,int *status,int Nx,int Ny)
{
    int k=0;
    int nBytes = npts*sizeof(double);
    int nBytes_fl = npts*sizeof(float);
    double rn, rn0;
    
    //HOST
    double *x,*b,*r,*z;
    float *r_fl,*z_fl;
    
    x=(double*)malloc(nBytes);
    b=(double*)malloc(nBytes);
    r=(double*)malloc(nBytes);
    z=(double*)malloc(nBytes);
    r_fl=(float*)malloc(nBytes_fl);
    z_fl=(float*)malloc(nBytes_fl);
    
     
    memset(x,0,nBytes);
    memset(b,0,nBytes);
    memset(r,0,nBytes);
    memset(z,0,nBytes);
    memset(r_fl,0,nBytes_fl);
    memset(z_fl,0,nBytes_fl);
    memset(x_fl,0,nBytes_fl);
    
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
    
    fl2d(b,b_fl);
    rn=norm_sqr_fl(b_fl);
    rn0=rn;
    status[0]=0;

    if (rn==0.0)
      return rn; 
    
       fl2d(x,x_fl);
       CHECK(cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice));
       CHECK(cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice));

    k=1;
    while(k<maxiter)
    {
       assign_v2v_gpu<<<grid,block>>>(d_xold,d_x,Nx,Ny);
        
       //r_k <- b-A_x_(k-1)  (DOUBLE PRECISION)
       laplace_2d_gpu<<<grid,block>>>(d_s,d_xold,Nx,Ny);
       add_gpu<<<grid,block>>>(d_r,d_b,-1.0,d_s,Nx,Ny);
       CHECK(cudaMemcpy(r, d_r, nBytes, cudaMemcpyDeviceToHost)); 
       rn=norm_sqr(r);
       
       if ((rn/rn0)<=rel)
        {
         break;
        } 
        
       //z_k <- A/r_k  (SINGLE PRECISION)
       d2fl(r_fl,r);
       cg_gpu_pc_mp(z_fl,r_fl,maxiter,rel,status,Nx,Ny,nBytes_fl);

       //x_k <- x_(k-1)+z_k  (DOUBLE PRECISION)
       fl2d(z,z_fl);
       CHECK(cudaMemcpy(d_z, z, nBytes, cudaMemcpyHostToDevice));
       add_gpu<<<grid,block>>>(d_x,d_xold,1.0,d_z,Nx,Ny);
    
       k++; 
    }

    
#ifdef DEBUG
   printf("Rel. Residuumnorm nach %d Iterationen: %e\n",k,sqrt(rn/rn0));
#endif 
    
   if ((rn/rn0<=rel) && (k<=maxiter))
      *status=k;
   if (rn/rn0>rel)
      *status=-1;
   
    
   CHECK(cudaMemcpy(x, d_xold, nBytes, cudaMemcpyDeviceToHost)); 

   free(b);
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
