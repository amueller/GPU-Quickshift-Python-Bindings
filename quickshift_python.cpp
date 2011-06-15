#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>

#include "Image.h"
#include "Exception.h"
//#include <fstream>
#include "quickshift_common.h"
#include <cutil_inline.h>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayGPUQuickshift
#include <numpy/arrayobject.h> // in python/lib/site-packages/....

using namespace boost::python;
using namespace std;

void image_from_data(image_t & im, float* image_data, int h, int w, int channels){
    im.I = image_data;
    im.N1 = h;
    im.N2 = w;
    im.K = channels;
    
}

int * map_to_flatmap(float * map, unsigned int size)
{
  /********** Flatmap **********/
  int *flatmap      = (int *) malloc(size*sizeof(int)) ;
  for (unsigned int p = 0; p < size; p++)
  {
    flatmap[p] = map[p];
  }

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (unsigned int p = 0; p < size; p++)
    {
      changed = changed || (flatmap[p] != flatmap[flatmap[p]]);
      flatmap[p] = flatmap[flatmap[p]];
    }
  }

  /* Consistency check */
  for (unsigned int p = 0; p < size; p++)
    assert(flatmap[p] == flatmap[flatmap[p]]);

  return flatmap;
}
image_t imseg(image_t im, int * flatmap)
{
  /********** Mean Color **********/
  float * meancolor = (float *) calloc(im.N1*im.N2*im.K, sizeof(float)) ;
  float * counts    = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

  for (int p = 0; p < im.N1*im.N2; p++)
  {
    counts[flatmap[p]]++;
    for (int k = 0; k < im.K; k++)
      meancolor[flatmap[p] + k*im.N1*im.N2] += im.I[p + k*im.N1*im.N2];
  }

  int roots = 0;
  for (int p = 0; p < im.N1*im.N2; p++)
  {
    if (flatmap[p] == p)
      roots++;
  }
  printf("Roots: %d\n", roots);

  int nonzero = 0;
  for (int p = 0; p < im.N1*im.N2; p++)
  {
    if (counts[p] > 0)
    {
      nonzero++;
      for (int k = 0; k < im.K; k++)
        meancolor[p + k*im.N1*im.N2] /= counts[p];
    }
  }
  if (roots != nonzero)
    printf("Nonzero: %d\n", nonzero);
  assert(roots == nonzero);


  /********** Create output image **********/
  image_t imout = im;
  imout.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
  for (int p = 0; p < im.N1*im.N2; p++)
    for (int k = 0; k < im.K; k++)
      imout.I[p + k*im.N1*im.N2] = meancolor[flatmap[p] + k*im.N1*im.N2];

  free(meancolor);
  free(counts);

  return imout;
}

PyObject * quickshift_python_wrapper(PyArrayObject image, float tau, float sigma, int* device=NULL){
    assert(image.nd == 2 || image.nd == 3);
    int* dims=new int[3];
    dims[2]=3;
    dims[1]=image.dimensions[1];
    dims[0]=image.dimensions[0];

    if(device!=NULL)
        cutilSafeCall(cudaSetDevice(*device));
    else
        cudaSetDevice(cutGetMaxGflopsDeviceId());

    float *map, *E, *gaps;
    int * flatmap;
    image_t imout;

    unsigned int totaltimer;
    cutilCheckError( cutCreateTimer(&totaltimer) );
    cutilCheckError( cutResetTimer(totaltimer) );
    cutilCheckError( cutStartTimer(totaltimer) );

    map          = (float *) calloc(dims[0]*dims[1], sizeof(float)) ;
    gaps         = (float *) calloc(dims[0]*dims[1], sizeof(float)) ;
    E            = (float *) calloc(dims[0]*dims[1], sizeof(float)) ;

    unsigned int timer=0;
    cutilCheckError( cutResetTimer(timer) );
    cutilCheckError( cutStartTimer(timer) );

    image_t im;
    image_from_data(im,(float*)image.data,image.dimensions[0],image.dimensions[1], image.dimensions[2]);

    /********** Quick shift **********/
    quickshift_gpu(im, sigma, tau, map, gaps, E);

    cutilCheckError( cutStopTimer(timer) );
    float modeTime = cutGetTimerValue(timer);

    /* Consistency check */
    for(int p = 0; p < im.N1*im.N2; p++)
        if(map[p] == p) assert(gaps[p] == INF);

    flatmap = map_to_flatmap(map, im.N1*im.N2);
    imout = imseg(im, flatmap);
    
    assert(imout.N1 == image.dimensions[0]);
    assert(imout.N2 == image.dimensions[1]);
    assert(imout.K == 3);
    PyArrayObject * out = (PyArrayObject*) PyArray_FromDimsAndData( 3, dims, PyArray_FLOAT, (char*)imout.I);

    free(flatmap);

    printf("Time: %fms\n\n\n", modeTime);

    cutilCheckError( cutStopTimer(totaltimer) );
    float totalTime = cutGetTimerValue(totaltimer);
    printf("Total time: %fms\n", totalTime);


    /********** Cleanup **********/
    free(im.I);

    free(map);
    free(E);
    free(gaps);
    delete dims;	
    return PyArray_Return(out);
}

BOOST_PYTHON_MODULE(quickshift_py){
    def("quickshift",quickshift_python_wrapper);
}
