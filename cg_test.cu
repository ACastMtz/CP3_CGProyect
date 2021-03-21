// Test function for CG
// Computes: ||r||_2 = ||b-Ax||_2 
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


double cg_test(double *a,double *b)
{
   int nBytes=npts*sizeof(double);
   double *s, *r, result;

   s=(double*)malloc(nBytes);
   r=(double*)malloc(nBytes);
   memset(s,0,nBytes);
   memset(r,0,nBytes);

   laplace_2d(s, a); 		
   add_mul_cpu(b,-1,s,r);	
   result=norm_sqr(r);	 

   free(s);
   free(r);

   return sqrt(result);
}
