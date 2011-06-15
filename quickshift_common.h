#ifndef __QUICKSHIFT_COMMON_H__
#define __QUICKSHIFT_COMMON_H__

#include <float.h>

typedef unsigned int vl_uint32 ;
typedef unsigned char vl_uint8 ; 
typedef unsigned short vl_uint16 ;

#define INF FLT_MAX

#define VL_MIN(a,b) ( ((a) <  (b) ) ? (a) : (b) )
#define VL_MAX(a,b) ( ((a) >  (b) ) ? (a) : (b) )
#define VL_ABS(a)   ( ((a) >= 0   ) ? (a) :-(a) )

typedef struct _image_t
{
  float * I;
  int N1,N2, K;
} image_t;

void quickshift(image_t im, float sigma, float tau, float * map, float * gaps, float * E);

extern "C" 
void quickshift_gpu(image_t im_d, float sigma, float tau_d, float * map_d, float * gaps_d, float * E_d);

#endif
