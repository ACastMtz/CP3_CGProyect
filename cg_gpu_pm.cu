//DEFINITION OF CG WITHOUT PRECONDITIONING (DOUBLE PRECISION) AND NO PRINTS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"
#include "common.h"

#define DEBUG

//GPU VERSION

double cg_gpu_pm(double *x,double *r,int maxiter,double rel,int *status,int Nx,int Ny,int nBytes)
{
   // CPU
   int k;
   double beta,rn,rnold,rn0,alpha_gpu;
   double *p,*s,*qr,*qs;
   
   p=(double*)malloc(npts*sizeof(double));
   s=(double*)malloc(npts*sizeof(double));
   qr=(double*)malloc(sizeof(double));
   qs=(double*)malloc(sizeof(double));
   
    
   memset(p,0,npts*sizeof(double));
   memset(s,0,npts*sizeof(double));
   memset(x,0,npts*sizeof(double));
   memset(qr,0,sizeof(double));
   memset(qs,0,sizeof(double));
   
   // GPU  
   double *d_p,*d_s,*d_x,*d_r,*d_qr,*d_qs;

   CHECK(cudaMalloc((void **)&d_p, nBytes));
   CHECK(cudaMalloc((void **)&d_s, nBytes)); 
   CHECK(cudaMalloc((void **)&d_x, nBytes));
   CHECK(cudaMalloc((void **)&d_r, nBytes));
   CHECK(cudaMalloc((void **)&d_qr, sizeof(double)));
   CHECK(cudaMalloc((void **)&d_qs, sizeof(double)));
    
   CHECK(cudaMemset(d_s,0,nBytes));
   CHECK(cudaMemset(d_p,0,nBytes));
   CHECK(cudaMemset(d_x,0,nBytes));
   CHECK(cudaMemset(d_qr,0,sizeof(double)));
   CHECK(cudaMemset(d_qs,0,sizeof(double)));
   
   int Nunroll=8;
   dim3 block2 (256,1);  
   int nblk = (npts + (block2.x*Nunroll) - 1)/(block2.x*Nunroll);
   dim3 grid2 (nblk,1);   
    
   // transfer data from host to device
   // r <- b
   CHECK(cudaMemcpy(d_r, r, nBytes, cudaMemcpyHostToDevice));
   rn=norm_sqr(r);
    
   rn0=rn;
   status[0]=0;
#ifdef DEBUG
   //printf("Residuumnorm am Anfang: %e\n",sqrt(rn0));
#endif
   if (rn==0.0)
      return rn;
   
   assign_v2v_gpu<<<grid,block>>>(d_p,d_r,Nx,Ny);       
   rel*=rel;
   k=0;

    
   while (k<maxiter)
   {
      
      laplace_2d_gpu<<<grid,block>>>(d_s,d_p,Nx,Ny);     // q <- A*d

      //Invoke dot_product_kernel
      dot_product_kernel<<<grid2,block2>>>(d_p,d_r,d_qr,npts);
      dot_product_kernel<<<grid2,block2>>>(d_p,d_s,d_qs,npts);
      
      //Device -> Host for division, not N-dependent 
      cudaMemcpy(qr,d_qr,sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(qs,d_qs,sizeof(double),cudaMemcpyDeviceToHost);
      alpha_gpu=(*qr)/(*qs);
             
      mul_add_gpu<<<grid,block>>>(d_x,alpha_gpu,d_p,Nx,Ny);  // x <- x+al*d
      mul_add_gpu<<<grid,block>>>(d_r,-alpha_gpu,d_s,Nx,Ny); // r <- r-al*q
      rnold=rn;           // delta_old <- delta_new
      CHECK(cudaMemcpy(r, d_r, nBytes, cudaMemcpyDeviceToHost)); 
      rn=norm_sqr(r);
      k+=1;
#ifdef DEBUG
      if (k % 100 == 0)
      {
         //printf("Iter %d, rel. Residuumnorm: %e\n",k,sqrt(rn/rn0));
      }
#endif
      if ((rn/rn0)<rel)
      {
         break;
      }
      
      beta=rn/rnold;
      update_p_gpu<<<grid,block>>>(d_r,beta,d_p,Nx,Ny);
      
      CHECK(cudaMemset(d_qr, 0.0, sizeof(double)));
      CHECK(cudaMemset(d_qs, 0.0, sizeof(double)));
   }

#ifdef DEBUG
   //printf("Rel. Residuumnorm nach %d Iterationen: %e\n",k,sqrt(rn/rn0));
#endif

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

   return sqrt(rn);
}
