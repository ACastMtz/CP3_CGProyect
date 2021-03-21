//DEFINITION OF POWER METHOD(CPU VERSION)
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
#include "cg_gpu_pm.h"


void pow_method_gpu(double *w,double *vl,double *lamb_max_gpu,double *lamb_min_gpu,int maxiter,int *status,int Nx,int Ny)
{
    int k_max,k_min;
    int nBytes=npts*sizeof(double);
    double *v, *s, *wl;
    double iNorm, l_old, err;
    
    int Nunroll=8;
    dim3 block2 (256,1);  
    int nblk = (npts + (block2.x*Nunroll) - 1)/(block2.x*Nunroll);
    dim3 grid2 (nblk,1);  
    
    v=(double*)malloc(sizeof(double));
    s=(double*)malloc(nBytes);
    wl=(double*)malloc(nBytes);
    
    memset(v,0, sizeof(double));
    memset(s,0,nBytes);
    memset(wl,0,nBytes);
    
    double *d_v, *d_w, *d_z, *d_wint, *d_l, *d_wl;
    
    CHECK(cudaMalloc((void **)&d_v, nBytes));
    CHECK(cudaMalloc((void **)&d_w, nBytes));
    CHECK(cudaMalloc((void **)&d_z, sizeof(double)));
    CHECK(cudaMalloc((void **)&d_wint, nBytes));
    CHECK(cudaMalloc((void **)&d_l, sizeof(double)));
    CHECK(cudaMalloc((void **)&d_wl, nBytes));
    
    CHECK(cudaMemset(d_v,0,nBytes));
    CHECK(cudaMemset(d_w,0,nBytes));
    CHECK(cudaMemset(d_z,0,sizeof(double)));
    CHECK(cudaMemset(d_v,0,nBytes));
    CHECK(cudaMemset(d_l,0,sizeof(double)));
    CHECK(cudaMemset(d_wl,0,nBytes));
    
    //Normalizing input array
    iNorm=norm_sqr(w);
    iNorm=1/sqrt(iNorm);
    mul_cpu(w,iNorm,w);
    
    CHECK(cudaMemcpy(d_w, w, nBytes, cudaMemcpyHostToDevice));

    
    k_max=0;
    l_old = 0.0;
    while(k_max<maxiter)
    {
        //Maximum eigenvalue
        
        //STEP: z_k <- A*q_(k-1)
        laplace_2d_gpu<<<grid,block>>>(d_v,d_w,Nx,Ny); 
        
        //STEP: q_k <- z_k/||z_k||
        dot_product_kernel<<<grid2,block2>>>(d_v, d_v, d_z, npts);
        CHECK(cudaMemcpy(v,d_z,sizeof(double),cudaMemcpyDeviceToHost)); 
        iNorm = 1/sqrt(*v);
        precondition_gpu<<<grid,block>>>(d_w,iNorm,d_v,Nx,Ny);
        
        //STEP: lambda_k <- q_k^TAq_k
        laplace_2d_gpu<<<grid,block>>>(d_wint,d_w,Nx,Ny); 
        dot_product_kernel<<<grid2,block2>>>(d_w, d_wint, d_l, npts);
        CHECK(cudaMemcpy(lamb_max_gpu,d_l,sizeof(double),cudaMemcpyDeviceToHost)); 
     
        k_max++;
       
        //Error
        err = abs((l_old-(*lamb_max_gpu))/(*lamb_max_gpu));
        l_old = *lamb_max_gpu;
        if (err<1e-8 && k_max>1)
          {
            break;
          }
        
        CHECK(cudaMemset(d_z, 0.0,sizeof(double)));
        CHECK(cudaMemset(d_l, 0.0,sizeof(double)));
    
    } 
    printf("After %d iterations the error for l_max is: %e\n",k_max,err);
    
    CHECK(cudaMemset(d_z, 0.0,sizeof(double)));
    CHECK(cudaMemset(d_l, 0.0,sizeof(double)));
    k_min=0; 
    l_old = 0.0;
    err = 0.0;
    
    while(k_min<maxiter)
    {
        
      //Minimum eigenvalue
        
        //STEP: A*q_k -> z_(k-1)  
        cg_gpu_pm(wl,vl,maxiter,1e-10,status,Nx,Ny,nBytes);
        
        //STEP: z_k <- q_k/||q_k||
        iNorm=norm_sqr(wl);
        iNorm=1/sqrt(iNorm);
        mul_cpu(vl,iNorm,wl);
        
        
        //STEP: lambda_k <- 1/(z_k^T.z_k)
        *lamb_min_gpu = iNorm;
        
        k_min++;
        
        //Error
        err = abs((l_old-(*lamb_min_gpu))/(*lamb_min_gpu));
        l_old = *lamb_min_gpu;
        if (err<1e-8 && k_min>1)
          {
            break;
          }
        
        CHECK(cudaMemset(d_z, 0.0,sizeof(double)));
        CHECK(cudaMemset(d_l, 0.0,sizeof(double)));
    }   
    printf("After %d iterations the error for l_min is: %e\n",k_min,err);
    
   free(v);
   free(s);
   free(wl);
    
   cudaFree(d_v);
   cudaFree(d_w);
   cudaFree(d_z);
   cudaFree(d_wint);
   cudaFree(d_l);
   cudaFree(d_wl);
    
}