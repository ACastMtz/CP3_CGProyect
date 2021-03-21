#ifndef LINALG_H
#define LINALG_H

extern void random_vector(double *p);
extern double norm_sqr(double *v);
extern double vector_prod(double *v, double *w);
extern void assign_v2v(double *v, double *w);
extern void mul_add(double *v, double a, double *w);
extern void update_p(double *r, double b, double *p);
extern void laplace_2d(double *w, double *v);
extern double norm_sqr_fl(float *v);
extern void add_mul_cpu(double *v, double a, double *w, double *r);
extern void mul_cpu(double *v, double a, double *w);

extern __global__ void reduceUnrolling (double *g_idata, double *g_odata, unsigned int n);
extern __global__ void assign_v2v_gpu(double *v, double *w, int nx, int ny);
extern __global__ void mul_add_gpu(double *v, double a, double *w, int nx, int ny);
extern __global__ void update_p_gpu(double *r, double b, double *p, int nx, int ny);
extern __global__ void laplace_2d_gpu(double *w, double *v, int nx, int ny);
extern __global__ void precondition_gpu(double *w, double b, double *v, int nx, int ny);
extern __device__ double atomicAdd_double(double* address, double val);
extern __global__ void dot_product_kernel(double *x, double *y, double *dot, unsigned int n);
extern __global__ void add_gpu(double *sol,double *w, double b, double *v, int nx, int ny);
extern __global__ void assign_v2v_gpu_fl(float *v, float *w, int nx, int ny);
extern __global__ void mul_add_gpu_fl(float *v, float a, float *w, int nx, int ny);
extern __global__ void update_p_gpu_fl(float *r, float b, float *p, int nx, int ny);
extern __global__ void laplace_2d_gpu_fl(float *w, float *v, int nx, int ny);
extern __global__ void precondition_gpu_fl(float *w, float b, float *v, int nx, int ny);
extern __global__ void dot_product_kernel_fl(float *x, float *y, float *dot, unsigned int n);

#endif
