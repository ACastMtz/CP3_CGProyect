/*********************************************************************
linalg.cu

Funktionen die Vektor-Operationen und Matrix-Vektor-Opertionen implementieren:

void random_vector(double *p)
   Setzt p im Inneren auf zufaellige Werte

__global__ void reduceUnrolling (double *g_idata, double *g_odata, unsigned int n)
   Reduktion auf der GPU. Berechnet die Teilsumme auf jedem Block mit vorheriger
   maximaler Anzahl von 'seriellen' Additionen pro Thread.

double norm_sqr(double *v)
   Quadrat der Norm von v.

double vector_prod(double *v, double *w)
   Vektorprodukt v^T * w

void assign_v2v(double *v, double *w)
__global__ void assign_v2v_gpu(double *v, double *w, int nx, int ny)
   Zuweisung v = w

void mul_add(double *v, double a, double *w)
__global__ void mul_add_gpu(double *v, double a, double *w, int nx, int ny)
   Multiplikation von w mit Skalar a und Addition zu v. Ergebnis in v.
                  v = v + a*w

void update_p(double *r, double b, double *p)
__global__ void update_p_gpu(double *r, double b, double *p, int nx, int ny)
   Multiplikation von p mit Skalar b und Addition zu r. Ergebnis in p.
                  p = r + b*p

void laplace_2d(double *w, double *v)
__global__ void laplace_2d_gpu(double *w, double *v, int nx, int ny)
   Anwendung des 2-D Laplace-Operator A=-\Delta mit Dirichlet-Randbedingungen
                  w = A*v

__global__ void precondition_gpu(double *w, double b, double *v, int nx, int ny)
  Kern function that multiplies vector v by a escalar and saves the result in vector w.
			 w = b*v

__device__ double atomicAdd_double(double* address, double val)
  Device function that allows the atomicAdd function to be implemented with double precission

__global__ void dot_product_kernel(double *x, double *y, double *dot, unsigned int n)
  Kernel function that implements the dot product of two vectors x and y. Result is saved in dot.
			  dot = x.y

__global__ void add_gpu(double *sol,double *w, double b, double *v, int nx, int ny)
  Kernel function that adds two vectors w and v, one of them multiplied by a scalar b. Result saved in sol.
			sol = w + b*v

//
The following functions were already described above, the difference being that they are redifined to work on single precission:

double norm_sqr_fl(float *v)

__global__ void assign_v2v_gpu_fl(float *v, float *w, int nx, int ny)

__global__ void mul_add_gpu_fl(float *v, float a, float *w, int nx, int ny)

__global__ void update_p_gpu_fl(float *r, float b, float *p, int nx, int ny)

__global__ void laplace_2d_gpu_fl(float *w, float *v, int nx, int ny)

__global__ void precondition_gpu_fl(float *w, float b, float *v, int nx, int ny)

__global__ void dot_product_kernel_fl(float *x, float *y, float *dot, unsigned int n)
//

void add_mul_cpu(double *v, double a, double *w, double *r) 
  Function that multiplies vector w with scalar a, adds it to v and stores the result in r
			r = v + a*w

void mul_cpu(double *v, double a, double *w)
  Function that multiplies the vector w with a scalar a and stores the result in v
			v = a*w

**********************************************************************/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <ctime>
#include "global.h"
#include "geometry.h"
#include "linalg.h"
#include "common.h"
#include "transform.h"

/*
   Der Vektor p wird im Inneren auf zufaellige Werte gesetzt
*/
void random_vector(double *p)
{
   int idx;

   for(idx = 0; idx < npts; idx++)
   {
      if (active[idx])
         p[idx] = (double)(rand() & 0xFF ) / 10.0;
   }
}

__global__ void reduceUnrolling (double *g_idata, double *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // unroll as many as possible
    unsigned int nunroll=n/gridSize, k;
    unsigned int i=idx+nunroll*gridSize;
    double sum=0.0;
    if (i<n)
        sum += g_idata[i];
    if (i+blockDim.x<n)
        sum += g_idata[i+blockDim.x];
    for (k=1; k<=nunroll; k++)
    {
        i -= gridSize;
        sum += g_idata[i] + g_idata[i+blockDim.x];
    }
    g_idata[idx] = sum;

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            g_idata[idx] += g_idata[idx + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = g_idata[idx];
}

double norm_sqr(double *v)
{
   int idx;
   double r=0.0;
   for (idx=0; idx<npts; idx++)
   {
      r+=v[idx]*v[idx];
   }
   return r;
}

double vector_prod(double *v, double *w)
{
   int idx;
   double r=0.0;
   for (idx=0; idx<npts; idx++)
   {
      r+=v[idx]*w[idx];
   }
   return r;
}

void assign_v2v(double *v, double *w)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      v[idx]=w[idx];
   }
}

__global__ void assign_v2v_gpu(double *v, double *w, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      v[idx]=w[idx];
   }
}

void mul_add(double *v, double a, double *w)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      v[idx]+=a*w[idx];
   }
}

__global__ void mul_add_gpu(double *v, double a, double *w, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      v[idx]+=a*w[idx];
   }
}

void update_p(double *r, double b, double *p)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      p[idx]=r[idx]+b*p[idx];
   }
}

__global__ void update_p_gpu(double *r, double b, double *p, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      p[idx]=r[idx]+b*p[idx];
   }
}

/*
   2D Laplace-Operator A=-\Delta, Dirichlet-Randbedingungen,
   multipliziert mit Vektor v:

                  w = A*v

   Wirkt nur auf innere/aktive Punkte. Aeussere Punkte bleiben unveraendert.
*/
void laplace_2d(double *w, double *v)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      if (active[idx])
         w[idx]=4.0*v[idx] - v[idx+1] - v[idx-1] - v[idx+Nx+2] - v[idx-Nx-2];
   }
}

__global__ void laplace_2d_gpu(double *w, double *v, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      w[idx]=4.0*v[idx] - v[idx+1] - v[idx-1] - v[idx+nx+2] - v[idx-nx-2];
   }
}

__global__ void precondition_gpu(double *w, double b, double *v, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      w[idx]=b*v[idx];
   }
}

__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void dot_product_kernel(double *x, double *y, double *dot, unsigned int n)
{
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x+1;
    unsigned int stride = blockDim.x*gridDim.x;
    
    __shared__ double cache[256];
    
    double temp = 0.0;
    while(index < n)
    {
        temp += x[index]*y[index];
        index += stride;
    }
    
    cache[threadIdx.x] = temp;
    
    __syncthreads();
    
    //Reduction
    unsigned int i = blockDim.x/2;
    while(i != 0)
    {
        if(threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if(threadIdx.x == 0)
    {
        atomicAdd_double(dot, cache[0]);
    }
}

__global__ void add_gpu(double *sol,double *w, double b, double *v, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      sol[idx]=w[idx]+b*v[idx];
   }
}

double norm_sqr_fl(float *v)
{
   int idx;
   float r=0.0;
   for (idx=0; idx<npts; idx++)
   {
      r+=v[idx]*v[idx];
   }
   return r;
}

__global__ void assign_v2v_gpu_fl(float *v, float *w, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      v[idx]=w[idx];
   }
}

__global__ void mul_add_gpu_fl(float *v, float a, float *w, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      v[idx]+=a*w[idx];
   }
}

__global__ void update_p_gpu_fl(float *r, float b, float *p, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      p[idx]=r[idx]+b*p[idx];
   }
}

__global__ void laplace_2d_gpu_fl(float *w, float *v, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      w[idx]=4.0*v[idx] - v[idx+1] - v[idx-1] - v[idx+nx+2] - v[idx-nx-2];
   }
}


__global__ void precondition_gpu_fl(float *w, float b, float *v, int nx, int ny)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
   unsigned int idx = iy * (nx+2) + ix;

   if (ix<=nx && iy<=ny)
   {
      w[idx]=b*v[idx];
   }
}


__global__ void dot_product_kernel_fl(float *x, float *y, float *dot, unsigned int n)
{
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    
    __shared__ float cache[256];
    
    float temp = 0.0;
    while(index < n)
    {
        temp += x[index]*y[index];
        index += stride;
    }
    
    cache[threadIdx.x] = temp;
    
    __syncthreads();
    
    //Reduction
    unsigned int i = blockDim.x/2;
    while(i != 0)
    {
        if(threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if(threadIdx.x == 0)
    {
        atomicAdd(dot, cache[0]);
    }
}

void add_mul_cpu(double *v, double a, double *w, double *r)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      r[idx]=v[idx]+a*w[idx];
   }
}

void mul_cpu(double *v, double a, double *w)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      v[idx]=a*w[idx];
   }
}


