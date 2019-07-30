

#ifndef GLOBAL_H
#define GLOBAL_H

#define DBL_EPSILON 2.2204460492503131e-16

#ifdef MAIN_PROGRAM
   #define EXTERN
#else
   #define EXTERN extern
#endif

/*
   Globale Variablen stehen in allen Funktionen zur Verfuegung.
   Achtung: Das gilt *nicht* fuer Kernel-Funktionen!
*/
EXTERN int Nx, Ny, npts;
EXTERN int *active;
EXTERN dim3 block, grid;

#undef EXTERN

#endif
