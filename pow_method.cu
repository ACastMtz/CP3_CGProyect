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
#include "cg_pm.h"


void pow_method(double *w,double *v_l,double *lamb_max,double *lamb_min,int maxiter,int *status)
{
    int k_max,k_min;
    int nBytes=npts*sizeof(double);
    double *v, *w_l, *r;
    double iNorm, l_old, err;
    
    v=(double*)malloc(nBytes);
    r=(double*)malloc(nBytes);
    w_l=(double*)malloc(nBytes);
    
    memset(v,0,nBytes);
    memset(r,0,nBytes);
    memset(w_l,0,nBytes);
   
    assign_v2v(w_l,w);
    
    //Normalizing input array
    iNorm=norm_sqr(w);
    iNorm=1/sqrt(iNorm);
    mul_cpu(w,iNorm,w);
    
    
    k_max=0;
    l_old = 0.0;
    while(k_max<maxiter)
    {
        //Maximum eigenvalue
        
        //STEP: z_k <- A*q_(k-1)
        laplace_2d(v,w);
        
        //STEP: q_k <- z_k/||z_k|| 
        iNorm=norm_sqr(v);
        iNorm=1/sqrt(iNorm);
        mul_cpu(w,iNorm,v);
        
        //STEP: lambda_k <- q_k^TAq_k
        laplace_2d(r,w);
        *lamb_max=vector_prod(w,r);
               
        //Error
        err = abs((l_old-(*lamb_max))/(*lamb_max));
     
        if (err<1e-8 && k_max>1)
          {
            break;
          }
    
        l_old = *lamb_max;
        
        k_max++;
    } 
    printf("After %d iterations the error for l_max is: %e\n",k_max,err);
    
    k_min=0;
    l_old = 0.0;
    err = 0.0;
    while(k_min<maxiter)
    {
        //Minimum eigenvalue
        
        //STEP: q_k <- A^(-1)*z_(k-1)
        cg_pm(w_l,v_l,maxiter,1e-10,status);
        
        //STEP: q_k <- z_k/||z_k|| 
        iNorm=norm_sqr(w_l);
        iNorm=1/sqrt(iNorm);
        
        //STEP: lambda_k <- 1/(z_k^T.z_k)
        *lamb_min = iNorm;
        mul_cpu(v_l,iNorm,w_l);

        //Error
        err = abs((l_old-(*lamb_min))/(*lamb_min));
       
        if (err<1e-8 && k_min>1)
          {
            break;
          }
    
        l_old = *lamb_min;
        
        k_min++;
    }   
    printf("After %d iterations the error for l_min is: %e\n",k_min,err);
    
   free(v);
   free(r);
   free(w_l);
    
}