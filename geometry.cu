/*********************************************************************
geometry.cu

Definiert die Gittergeometrie, Ausgabe:

int coord2index(int i, int j)
   Berechnet den fortlaufenden Index fuer die Koordianten (i,j)

void active_pts()
   Initialisiert das globale Flag-Array der aktiven/inneren Punkte.

void print_active()
   Das Flag-Array der aktiven/inneren Punkte wird als 2-D Gitter ausgegeben.

void print_vector(const char *name, double *p, int flag)
   Der Vektor p wird als 2D Gitter fuer i,j<=16 ausgegeben. Es werden innere/aktive
   und, falls flag>0, auch die aeusseren Punkte ausgegeben.

**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"

/*
   Fuer die Koordinaten:
      i = 0,1,...,Nx+1
      j = 0,1,...,Ny+1
   wird der fortlaufenden Index berechnet
*/
int coord2index(int i, int j)
{
   return j*(Nx+2) + i;
}

/*
   Das Flag-Array der aktiven/inneren Punkte wird gesetzt.
*/
void active_pts()
{
   int idx,i,j;

   active=(int*)malloc(npts*sizeof(int));

   idx=0; // fortlaufender Index
   for (j=0; j<Ny+2; j++)
   {
      for (i=0; i<Nx+2; i++)
      {
         if ((i==0)||(j==0)||(i==Nx+1)||(j==Ny+1))
            active[idx]=0; // Randpunkt
         else
            active[idx]=1; // innerer Punkt

         idx+=1;
      }
   }
}

/*
   Das Flag-Array der aktiven/inneren Punkte wird als
   2D Gitter ausgegeben.
*/
void print_active()
{
   int i,j,idx;

   printf("active points:\n");
   idx=0;
   for (j=0; j<Ny+2; j++)
   {
      printf("  ");
      for (i=0; i<Nx+2; i++)
      {
         printf("%d ",active[idx]);
         idx+=1;
      }
      printf("\n");
   }
}

/*
   Der Vektor p wird als 2D Gitter fuer i,j<=16 ausgegeben. Es werden innere/aktive
   und, falls flag>0, auch die aeusseren Punkte ausgegeben.
*/
void print_vector(const char *name, double *p, int flag)
{
   int i,j,idx;
   double nrm;

   printf("%s = \n",name);
   idx=0;
   for (j=0; j<Ny+2; j++)
   {
      if (j>16)
      {
         printf("  ...\n");
         break;
      }
      printf("  ");
      for (i=0; i<Nx+2; i++)
      {
         if ((i<16)&&((flag>0)||(active[idx])))
            printf("%.2f ",p[idx]);
         if (i==16)
            printf("...");
         idx+=1;
      }
      printf("\n");
   }
   nrm=norm_sqr(p);
   printf("||%s|| = %.8f\n",name,sqrt(nrm));
}
