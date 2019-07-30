/*********************************************************************
run-cg.cu

Hauptprogramm. Testet Reduktion und ruft cg auf.

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
#include "cg.h"

int main(int argc, char **argv)
{
   printf("%s Starting...\n", argv[0]);

   int nBytes, status, N;
   double *w, *v, *x;
   double iStart, iElaps;

   N=32;
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

   // Globale Variablen setzen:
   // Anzahl der Inneren Punkte in x- und y-Richtung
   Nx=N;
   Ny=N;
   // Gesamtanzahl der Gitterpunkte
   npts=(Nx+2)*(Ny+2);
   // Aktive Punkte - Array
   active_pts();

   // Speicherbedarf pro Vektor in Byte
   nBytes=npts*sizeof(double);

   // Speicher für Vektoren allozieren
   w=(double*)malloc(nBytes);
   v=(double*)malloc(nBytes);

   // auf Null setzen
   memset(w, 0, nBytes);
   memset(v, 0, nBytes);

   // Aktive Punkte ausgeben
   if ((Nx<=16)&&(Ny<=16))
      print_active();

   random_vector(w);
   random_vector(v);
   double *d_v, *d_w, *d_x;
   CHECK(cudaMalloc((void **)&d_v, nBytes));
   CHECK(cudaMalloc((void **)&d_w, nBytes));

   // transfer data from host to device
   CHECK(cudaMemcpy(d_v, v, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_w, w, nBytes, cudaMemcpyHostToDevice));
   // invoke kernel at host side
   block.x=dimx;
   block.y=dimy;
   block.z=1;
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

   // Einheitsvektor
   memset(v, 0, nBytes);
   v[coord2index(Nx/2,Nx/2)]=1.0; // v=0, ausser am Gitterpunkt (Nx/2+1,Ny/2+1)
   print_vector("v",v,1);

   cg(w,v,1000,1e-10,&status);

   print_vector("x",w,0);

   free(active);
   free(w);
   free(v);

   return (0);
}
