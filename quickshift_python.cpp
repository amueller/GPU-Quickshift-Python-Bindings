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

void image_from_data(image_t & im, unsigned char* image_data, int N1, int N2, int channels){
    im.N1 = N1;
    im.N2 = N2;
    im.K = channels;

    im.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
    for(int k = 0; k < im.K; k++)
        for(int col = 0; col < im.N2; col++)
            for(int row = 0; row < im.N1; row++)
            {
                unsigned char * pt = image_data + im.K * (col + im.N2 *(im.N1-1-row));
                im.I[row + col*im.N1 + k*im.N1*im.N2] = 32. * pt[k] / 255.; // Scale 0-32
            }
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

//int quickshift_python_wrapper(PyArrayObject image, float tau, float sigma, int device){
PyObject * quickshift_python_wrapper(PyArrayObject image, float tau, float sigma, int device){

    //assert(image.nd == 2 || image.nd == 3);
    assert(image.nd == 3);
    cout << "Input needs to be scaled between 0 and 32!" <<std::endl;
    if (PyArray_TYPE(&image) != PyArray_UBYTE){
       cout << "Only float arrays are supported"  <<std::endl;
       exit(1);
    }
    
    npy_intp* dims=new npy_intp[3];
    dims[2]=3;
    dims[1]=image.dimensions[1];
    dims[0]=image.dimensions[0];

    cutilSafeCall(cudaSetDevice(device));

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

    unsigned char* blub = (unsigned char*)image.data;
    std::cout << int(blub[0]) << " " << int(blub[1]) << " " << int(blub[2]) <<std::endl;
    std::cout << int(blub[3* (30 + 20*dims[1])]) <<std::endl;

    image_t im;
    image_from_data(im,(unsigned char*)image.data,image.dimensions[0],image.dimensions[1], image.dimensions[2]);
    cout << im.I[0] << " " << im.I[1] << " " << im.I[2] <<endl;

    //[>********* Quick shift *********<]
    quickshift_gpu(im, sigma, tau, map, gaps, E);

    //[> Consistency check <]
    for(int p = 0; p < im.N1*im.N2; p++)
        if(map[p] == p) assert(gaps[p] == INF);

    flatmap = map_to_flatmap(map, im.N1*im.N2);
    imout = imseg(im, flatmap);
    
    assert(imout.N1 == image.dimensions[0]);
    assert(imout.N2 == image.dimensions[1]);
    assert(imout.K == 3);
    PyArrayObject * out = (PyArrayObject*) PyArray_SimpleNewFromData( 3, dims, PyArray_FLOAT, imout.I);
    //PyArrayObject * out = (PyArrayObject*) PyArray_SimpleNew( 3, dims, PyArray_FLOAT);

    free(flatmap);



    /********** Cleanup **********/
    //free(im.I);

    free(map);
    free(E);
    free(gaps);
    delete dims;	
    return PyArray_Return(out);
    //return PyArray_Return(&image);
}

void* extract_pyarray(PyObject* x)
{
	return x;
}

BOOST_PYTHON_MODULE(quickshift_py){
	converter::registry::insert(
	    &extract_pyarray, type_id<PyArrayObject>());
    def("quickshift",quickshift_python_wrapper);
    import_array();
}
