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

void fl2d(double *w, float *v)
{
   for(int i=0; i<npts; i++)
   {
    w[i] = (double) v[i];
   }
}

void d2fl(float *w, double *v)
{
   for(int i=0; i<npts; i++)
   {
    w[i] = (float) v[i];
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

void mul_cpu(double *v, double a, double *w)
{
   int idx;
   for (idx=0; idx<npts; idx++)
   {
      v[idx]=a*w[idx];
   }
}




