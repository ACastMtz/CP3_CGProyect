#ifndef LINALG_H
#define LINALG_H

extern void random_vector(double *p);
extern double norm_sqr(double *v);
extern double vector_prod(double *v, double *w);
extern void assign_v2v(double *v, double *w);
extern void mul_add(double *v, double a, double *w);
extern void update_p(double *r, double b, double *p);
extern void laplace_2d(double *w, double *v);

extern __global__ void reduceUnrolling (double *g_idata, double *g_odata, unsigned int n);
extern __global__ void assign_v2v_gpu(double *v, double *w, int nx, int ny);
extern __global__ void mul_add_gpu(double *v, double a, double *w, int nx, int ny);
extern __global__ void update_p_gpu(double *r, double b, double *p, int nx, int ny);
extern __global__ void laplace_2d_gpu(double *w, double *v, int nx, int ny);

#endif
