//DEFINITION OF CG AND NO PRINTS USED IN THE POWER ITERATION
//CPU VERSION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"
#include "common.h"


double cg_pm(double *x, double *r, int maxiter, double rel, int *status)
{
   int k;
   double ar,as,alpha,beta,rn,rnold,rn0;
   double *p,*s;

   s=(double*)malloc(npts*sizeof(double));
   p=(double*)malloc(npts*sizeof(double));

   memset(x,0,npts*sizeof(double));
   memset(s,0,npts*sizeof(double));

   rn=norm_sqr(r);
   rn0=rn;
   status[0]=0;

   if (rn==0.0)
      return rn;

   assign_v2v(p,r);
   rel*=rel;
   k=0;

   while (k<maxiter)
   {
      laplace_2d(s,p);
      ar=vector_prod(p,r);
      as=vector_prod(p,s);
      alpha=ar/as;
      mul_add(x,alpha,p);
      mul_add(r,-alpha,s);
      rnold=rn;
      rn=norm_sqr(r);
      k+=1;

      if ((rn/rn0)<rel)
      {
         break;
      }
      beta=rn/rnold;
      update_p(r,beta,p);
   }


   if ((rn/rn0<=rel) && (k<=maxiter))
      *status=k;
   if (rn/rn0>rel)
      *status=-1;

   free(s);
   free(p);

   return sqrt(rn);
}
