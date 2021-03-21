/*********************************************************************
transform.cu

Functions to transform single precission into double precission and vice-versa.

void fl2d(double *w, float *v)
  single to double precission

void d2fl(float *w, double *v)
  double to single precission


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
