/*********************************************************************
run-cg.cu

Main program. Computes the all the versions of CG and tests each
module. 

**********************************************************************/
#define MAIN_PROGRAM

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"
#include "transform.h"
#include "cg.h"
#include "cg_gpu.h"
#include "cg_gpu_sp.h"
#include "cg_gpu_mp.h"
#include "mp_refinement.h"
#include "mp_refinement_cpu.h"
#include "cg_pm.h"
#include "cg_gpu_pm.h"
#include "pow_method.h"
#include "pow_method_gpu.h"
#include "dot_prod_test.h"
#include "cg_test.h"
#include "pm_test.h"



int main(int argc, char **argv)
{
   printf("%s Starting...\n", argv[0]);
    
   int nBytes, nBytes_fl, status, N;
   double *w,*w_cpu,*w_gpu,*v,*v_cpu,*v_gpu,*x,*w_fl_d;
   double *w_pm,*lamb_max,*lamb_min,*v_pm, *w_pm_gpu, *v_pm_gpu,*lamb_max_gpu,*lamb_min_gpu;
   float *w_fl, *v_fl, *w_fl_mp, *v_fl_mp;
   double iStart, iElaps;

   N=256;
   int dimx = 256;
   int dimy = 1;
   if (argc>1)
   {
      N=atoi(argv[1]);
   }
   if (argc>3)
   {
      dimx=atoi(argv[2]);
      dimy=atoi(argv[3]);
   }

   // set up device
   int dev = 0;
   cudaDeviceProp deviceProp;
   CHECK(cudaGetDeviceProperties(&deviceProp, dev));
   printf("Using Device %d: %s\n", dev, deviceProp.name);
   CHECK(cudaSetDevice(dev));

   // Global variables
   // Inner points in x- and y-directions
   Nx=N;
   Ny=N;
   // Number of gridpoints
   npts=(Nx+2)*(Ny+2);
   // Active points array
   active_pts();

   // Memoryspace per vector in Byte
   nBytes=npts*sizeof(double);
   nBytes_fl=npts*sizeof(float);

   // Host allocation
   w=(double*)malloc(nBytes);
   v=(double*)malloc(nBytes);
   w_cpu=(double*)malloc(nBytes);
   v_cpu=(double*)malloc(nBytes);
   w_gpu=(double*)malloc(nBytes);
   v_gpu=(double*)malloc(nBytes);
   w_fl=(float*)malloc(nBytes_fl);
   v_fl=(float*)malloc(nBytes_fl);
   v_fl_mp=(float*)malloc(nBytes_fl);
   w_fl_mp=(float*)malloc(nBytes_fl);
   w_fl_d=(double*)malloc(nBytes);
   w_pm=(double*)malloc(nBytes);
   lamb_max=(double*)malloc(nBytes);
   lamb_min=(double*)malloc(nBytes);
   v_pm=(double*)malloc(nBytes);
   w_pm_gpu=(double*)malloc(nBytes);
   v_pm_gpu=(double*)malloc(nBytes);
   lamb_max_gpu=(double*)malloc(nBytes);
   lamb_min_gpu=(double*)malloc(nBytes);
    
   // Setting to zero
   memset(w, 0, nBytes);
   memset(v, 0, nBytes);
   memset(w_cpu, 0, nBytes);
   memset(v_cpu, 0, nBytes);
   memset(w_gpu, 0, nBytes);
   memset(v_gpu, 0, nBytes);
   memset(w_fl, 0, nBytes_fl);
   memset(v_fl, 0, nBytes_fl);
   memset(w_fl_mp, 0, nBytes_fl);
   memset(v_fl_mp, 0, nBytes_fl);
   memset(w_fl_d, 0, nBytes);
   memset(w_pm, 0, nBytes);
   memset(lamb_max, 0, nBytes);
   memset(lamb_min, 0, nBytes);
   memset(v_pm, 0, nBytes);
   memset(w_pm_gpu, 0, nBytes);
   memset(v_pm_gpu, 0, nBytes);
   memset(lamb_max_gpu, 0, nBytes);
   memset(lamb_min_gpu, 0, nBytes);
    
   // Active points
   if ((Nx<=16)&&(Ny<=16))
      print_active();

   random_vector(w);
   random_vector(v);
   assign_v2v(w_cpu,w);
   assign_v2v(w_gpu,w);
   assign_v2v(w_pm,w);
   assign_v2v(w_pm_gpu,w);
   
   
   // Device allocation
   double *d_v, *d_w, *d_x;
    
   CHECK(cudaMalloc((void **)&d_v, nBytes));
   CHECK(cudaMalloc((void **)&d_w, nBytes));
  
   // transfer data from host to device
   CHECK(cudaMemcpy(d_v, v, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_w, w, nBytes, cudaMemcpyHostToDevice));
   
   // invoke kernel at host side
   // Threads per block
   block.x=dimx;
   block.y=dimy;
   block.z=1;
   // Blocks per grid
   grid.x=(Nx + block.x - 1) / block.x;
   grid.y=(Ny + block.y - 1) / block.y;
   grid.z=1;
    
 
   // Test reduction
   int Nunroll=8;
   if (npts>256 && Nunroll>1)
   {
      double cpu_sum=0.0;
      iStart = seconds();
      for (int i = 0; i < npts; i++) cpu_sum += v[i];
      iElaps = seconds() - iStart;
      printf("cpu reduce      elapsed %f sec cpu_sum: %f\n", iElaps, cpu_sum);

      dim3 block2 (256,1);  
      int nblk = (npts + (block2.x*Nunroll) - 1)/(block2.x*Nunroll);
      dim3 grid2 (nblk,1);  
      CHECK(cudaMalloc((void **)&d_x, nblk*sizeof(double)));
      CHECK(cudaMemset(d_x,0,nblk*sizeof(double)));
      x=(double*)malloc(nblk*sizeof(double));
      CHECK(cudaDeviceSynchronize());
      iStart = seconds();
      reduceUnrolling<<<grid2, block2>>>(d_v, d_x, npts);
      CHECK(cudaDeviceSynchronize());
      iElaps = seconds() - iStart;
      CHECK(cudaMemcpy(x, d_x, nblk * sizeof(double),cudaMemcpyDeviceToHost));

      double gpu_sum = 0.0;
      for (int i = 0; i < grid2.x; i++) gpu_sum += x[i];

      printf("gpu Unrolling  elapsed %f sec gpu_sum: %f <<<grid %d block "
             "%d>>>\n", iElaps, gpu_sum, grid2.x, block2.x);

      assert(abs((gpu_sum-cpu_sum)/cpu_sum)<sqrt(npts)*DBL_EPSILON);
   }
  
   double nrm,nrm_mp;
   float nrm_fl;
    
   // Einheitsvektor
   memset(v, 0, nBytes);
   v[coord2index(Nx/2,Nx/2)]=1.0; // v=0, ausser am Gitterpunkt (Nx/2+1,Ny/2+1)
   //print_vector("v",v,1);
   assign_v2v(v_cpu,v);
   assign_v2v(v_gpu,v);
   assign_v2v(v_pm,v);
   assign_v2v(v_pm_gpu,v);
   
   d2fl(w_fl,w);
   d2fl(v_fl,v);
   d2fl(w_fl_mp,w);
   d2fl(v_fl_mp,v);   
   
   printf("\n"); 
   printf("N = %d\n", N);
   printf("\n");
     
   // CPU Version
   printf("CPU\n");
   iStart = seconds();
   cg(w_cpu,v_cpu,100000,1e-10,&status); 
   iElaps = seconds() - iStart;
   printf("CG on CPU: %f sec\n", iElaps);
   nrm=norm_sqr(w_cpu);
   printf("||x|| = %.8f\n",sqrt(nrm));
   printf("\n");
       
   // GPU Version
   printf("GPU\n");
   iStart = seconds();
   cg_gpu(w_gpu,v_gpu,100000,1e-10,&status, Nx, Ny, nBytes); 
   iElaps = seconds() - iStart;
   printf("CG on GPU: %f sec\n", iElaps);
   nrm=norm_sqr(w_gpu);
   printf("||x|| = %.8f\n",sqrt(nrm));
   printf("\n");

   // GPU Version in SP
   printf("GPU in single precission\n");
   iStart = seconds();
   cg_gpu_sp(w_fl,v_fl,100000,1e-08,&status, Nx, Ny, nBytes_fl); 
   iElaps = seconds() - iStart;
   printf("SP-CG on GPU: %f sec\n", iElaps);
   nrm_fl=norm_sqr_fl(w_fl);
   printf("||x|| = %.8f\n",sqrt(nrm_fl));
   printf("Rel. deviation of x = %e\n",abs((sqrt(nrm_fl)-sqrt(nrm))/sqrt(nrm)));
   printf("\n");
   
   double *w_mp_gpu, *v_mp_gpu;
   w_mp_gpu=(double*)malloc(nBytes);
   v_mp_gpu=(double*)malloc(nBytes);
   memset(w_mp_gpu, 0, nBytes);
   memset(v_mp_gpu, 0, nBytes); 
   v_mp_gpu[coord2index(Nx/2,Nx/2)]=1.0; 
   int iterMP = 1000;
    
   // Mixed precission
   printf("Initiating MP refinement (GPU)\n");   
   iStart = seconds();
   mp_refinement(w_mp_gpu,v_mp_gpu,100000,iterMP,1e-10,&status,Nx,Ny);
   iElaps = seconds() - iStart;
   printf("MP refinement: %f sec\n", iElaps);
   nrm_mp=norm_sqr(w_mp_gpu);
   printf("||x|| = %.8f\n",sqrt(nrm_mp));
   printf("Rel. deviation of x = %e\n",abs((sqrt(nrm_mp)-sqrt(nrm))/sqrt(nrm)));
   printf("\n");

   // Power iteration CPU
   printf("Power iteration CPU\n");
   iStart = seconds();
   pow_method(w_pm,v_pm,lamb_max,lamb_min,10000,&status);
   iElaps = seconds() - iStart;
   printf("PM on CPU: %f sec\n", iElaps);
   printf("Maximum eigenvalue l_max= %.8f\n",*lamb_max);
   printf("Minimum eigenvalue l_min= %.8f\n",*lamb_min);
   printf("Condition number = %.8f\n",(*lamb_max)/(*lamb_min));
   printf("\n");

   // Power iteration GPU
   printf("Power iteration GPU\n");
   iStart = seconds();
   pow_method_gpu(w_pm_gpu,v_pm_gpu,lamb_max_gpu,lamb_min_gpu,100000,&status,Nx,Ny);
   iElaps = seconds() - iStart;
   printf("PM on GPU: %f sec\n", iElaps);
   printf("Maximum eigenvalue l_max= %.8f\n",*lamb_max_gpu);
   printf("Minimum eigenvalue l_min= %.8f\n",*lamb_min_gpu);
   printf("Condition number = %.8f\n",(*lamb_max_gpu)/(*lamb_min_gpu));

   //-------------------Tests---------------------------------  
   
   //--Scalar product------//
    double *a,*b;
	
   a=(double*)malloc(nBytes);
   b=(double*)malloc(nBytes);
	
   random_vector(a);
   random_vector(b);
   printf("\n"); 
   printf("\n N = %d \n", N);
   dot_prod_test(a, b, nBytes);
   printf("\n"); 

   //--CG------------------// 
   double res=0.0;
   double *v_test, *w_test;
   v_test=(double*)malloc(nBytes);
   memset(v_test, 0, nBytes);
   w_test=(double*)malloc(nBytes);
   memset(w_test, 0, nBytes);
   fl2d(w_test,w_fl);

   printf("\n"); 
   printf("\n N = %d \n", N);
   printf("\n"); 

   v_test[coord2index(Nx/2,Nx/2)]=1.0; 
   res=cg_test(w_cpu,v_test);
   printf("\n Norm. residuum (CPU) ||r|| = ||b -Ax||: %e", res);
   v_test[coord2index(Nx/2,Nx/2)]=1.0; 
   res=cg_test(w_gpu,v_test);
   printf("\n Norm. residuum (GPU) ||r|| = ||b -Ax||: %e", res);
   v_test[coord2index(Nx/2,Nx/2)]=1.0; 
   res=cg_test(w_test,v_test);
   printf("\n Norm. residuum (SP-GPU) ||r|| = ||b -Ax||: %e", res);
   v_test[coord2index(Nx/2,Nx/2)]=1.0; 
   res=cg_test(w_mp_gpu,v_test);
   printf("\n Norm. residuum (MP-GPU) ||r|| = ||b -Ax||: %e", res);
   printf("\n"); 
   printf("\n N = %d \n", N);
   

   //----Power Method-----------//
   res=pm_test(w_pm,lamb_max);	 
   printf("\n Norm. residuum for lambda ||A x - lambda x||: %e", res);
   

   free(active);
   free(w);
   free(v);
   free(w_gpu);
   free(v_gpu);
   free(w_cpu);
   free(v_cpu);
   free(w_fl);
   free(v_fl);
   free(v_fl_mp);
   free(w_test);
   free(v_test);
   free(a);
   free(b);
   //free(res);
    
   cudaFree(d_w);
   cudaFree(d_v);
   cudaFree(d_x);

   return (0);
}