
// Test function for Power Method
// Computes: ||A q_k - lambda_k q_k||_2 
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


double pm_test(double *q,double *lamb)
{
   double *s,*p,*r;
   double nrm, result;
   int nBytes=npts*sizeof(double);

   s=(double*)malloc(nBytes);
   p=(double*)malloc(nBytes);
   r=(double*)malloc(nBytes);

   memset(s,0,nBytes);
   memset(p,0,nBytes);
   memset(r,0,nBytes);

   laplace_2d(s,q); 				
   mul_cpu(p,*lamb,q);			
   add_mul_cpu(s,-1,p,r);		
   nrm=norm_sqr(s);			
   result=norm_sqr(r);	

// Normalized result
   result=result/nrm;				

   free(s);
   free(p);

   return result;
}
