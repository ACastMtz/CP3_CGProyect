//DEFINITION OF CG (SINGLE PRECISION) FOR THE POWER ITERATION
// AND NO PRINTS
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


//GPU VERSION

double cg_gpu_mp(float *x,float *r,int maxiter,double rel,int *status,int Nx,int Ny,int nBytes)
{
   int k;
   double iterations=0;
   float beta,rn,rnold,rn0,alpha_gpu;
   
//CPU variables & initialization
   float *p,*s,*qr,*qs;
   
   p=(float*)malloc(npts*sizeof(float));
   s=(float*)malloc(npts*sizeof(float));
   qr=(float*)malloc(sizeof(float));
   qs=(float*)malloc(sizeof(float));

   memset(x,0,npts*sizeof(float));
   memset(p,0,npts*sizeof(float));
   memset(s,0,npts*sizeof(float));
   memset(qr,0,sizeof(float));
   memset(qs,0,sizeof(float));
 
//GPU variables & initialization 
   float *d_p,*d_s,*d_x,*d_r,*d_qr,*d_qs;

   CHECK(cudaMalloc((void **)&d_p, nBytes));
   CHECK(cudaMalloc((void **)&d_s, nBytes)); 
   CHECK(cudaMalloc((void **)&d_x, nBytes));
   CHECK(cudaMalloc((void **)&d_r, nBytes));
   CHECK(cudaMalloc((void **)&d_qr, sizeof(float)));
   CHECK(cudaMalloc((void **)&d_qs, sizeof(float)));
    
   CHECK(cudaMemset(d_s,0,nBytes));
   CHECK(cudaMemset(d_p,0,nBytes));
   CHECK(cudaMemset(d_x,0,nBytes));
   CHECK(cudaMemset(d_qr,0,sizeof(float)));
   CHECK(cudaMemset(d_qs,0,sizeof(float)));
    
//Execution configuration for dot-product-kernel  
   int Nunroll=8;
   dim3 block2 (256,1);  
   int nblk = (npts + (block2.x*Nunroll) - 1)/(block2.x*Nunroll);
   dim3 grid2 (nblk,1);   

   
  //STEP: r <- b  
   // Host -> Device
   CHECK(cudaMemcpy(d_r, r, nBytes, cudaMemcpyHostToDevice));
   // Initial residual (normalized) 
   rn=norm_sqr_fl(r);
   rn0=rn;
    
   status[0]=0;
   if (rn==0.0)
      return rn; 
    
   assign_v2v_gpu_fl<<<grid,block>>>(d_p,d_r,Nx,Ny);
    
   //Initial relative accuracy (^2)
   rel*=rel;
   
   k=0;
    
   while (k<maxiter)
   {
    //STEP: q <- A*d  
      laplace_2d_gpu_fl<<<grid,block>>>(d_s,d_p,Nx,Ny); 
    
   //STEP: delta_new <- r^T.s
      dot_product_kernel_fl<<<grid2,block2>>>(d_p,d_r,d_qr,npts);
   
    //STEP: d^T.q
      dot_product_kernel_fl<<<grid2,block2>>>(d_p,d_s,d_qs,npts);
    
    //STEP: alpha <- delta_new/d^T.q  
      //Device -> Host for division (N independent)
      CHECK(cudaMemcpy(qr,d_qr,sizeof(float),cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(qs,d_qs,sizeof(float),cudaMemcpyDeviceToHost));
      alpha_gpu=(*qr)/(*qs);
     
    //STEP: x <- x+alpha*d 
      mul_add_gpu_fl<<<grid,block>>>(d_x,alpha_gpu,d_p,Nx,Ny); 
    
    //STEP: r <- r-alpha*q
      mul_add_gpu_fl<<<grid,block>>>(d_r,-alpha_gpu,d_s,Nx,Ny);
     
    // delta_old <- delta_new 
	rnold=rn;              
    
    //Device -> Host for if-condition control (N independent)
CHECK(cudaMemcpy(r,d_r,nBytes,cudaMemcpyDeviceToHost));
      rn=norm_sqr_fl(r);
       
      k+=1;     

      if ((rn/rn0)<=rel)
      {
         break;
      }  
     
    //STEP: beta <- delta_new/delta_old
      beta = rn/rnold;
        
    //STEP: d <- s+beta*d
      update_p_gpu_fl<<<grid,block>>>(d_r,beta,d_p,Nx,Ny);
      
      //Cleaning intermediate variables
      CHECK(cudaMemset(d_qr, 0.0, sizeof(float)));
      CHECK(cudaMemset(d_qs, 0.0, sizeof(float)));
   }

   if ((rn/rn0<=rel) && (k<=maxiter))
      *status=k;
   if (rn/rn0>rel)
      *status=-1;
   
   CHECK(cudaMemcpy(x, d_x, nBytes, cudaMemcpyDeviceToHost)); 
 
   free(s);
   free(p);
   free(qr);
   free(qs);
    
   cudaFree(d_p);
   cudaFree(d_s);
   cudaFree(d_x);
   cudaFree(d_r);
   cudaFree(d_qr);
   cudaFree(d_qs);
	
   iterations = k;
   //return sqrt(rn);
   return iterations;

}


