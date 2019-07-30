/*********************************************************************
cg.cu

Conjugate Gradient

double cg(double *x, double *r, int maxiter, double rel, int *status)
   Loest lineares Gleichungssystem
                  A*x = r
   maxiter: Maximale Anzahl der Iterationen
   rel:     Relative Reduktion der Residuumnorm
   status:  Wird bei Erfolg auf Anzahl der Iterationen gesetzt. Sonst <=0.

   Rueckgabewert ist die erreichte Residuumnorm.

**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"


#define DEBUG

double cg(double *x, double *r, int maxiter, double rel, int *status)
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
#ifdef DEBUG
   printf("Residuumnorm am Anfang: %e\n",sqrt(rn0));
#endif
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
#ifdef DEBUG
      if (k % 10 == 0)
      {
         printf("Iter %d, rel. Residuumnorm: %e\n",k,sqrt(rn/rn0));
      }
#endif
      if ((rn/rn0)<rel)
      {
         break;
      }
      beta=rn/rnold;
      update_p(r,beta,p);
   }

#ifdef DEBUG
   printf("Rel. Residuumnorm nach %d Iterationen: %e\n",k,sqrt(rn/rn0));
#endif

   if ((rn/rn0<=rel) && (k<=maxiter))
      *status=k;
   if (rn/rn0>rel)
      *status=-1;

   free(s);
   free(p);

   return sqrt(rn);
}
