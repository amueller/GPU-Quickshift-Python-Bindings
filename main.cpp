#include "Image.h"
#include "Exception.h"
#include <fstream>
#include "quickshift_common.h"
#include <cutil_inline.h>

void write_image(image_t im, const char * filename)
{
  /********** Copy from matlab style **********/
  Image IMGOUT(im.K > 1 ? Image::RGB : Image::L, im.N2, im.N1);
  for(int k = 0; k < im.K; k++)
    for(int col = 0; col < im.N2; col++)
      for(int row = 0; row < im.N1; row++)
      {
        /* Row transpose */
        unsigned char * pt = IMGOUT.getPixelPt(col, im.N1-1-row);
        /* scale 0-255 */
        pt[k] = (unsigned char) (im.I[row + col*im.N1 + k*im.N1*im.N2]/32*255);
      }


  /********** Write image **********/
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
      throw Exception("Could not open the file");
  }
  ofs<<IMGOUT;
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

void image_to_matlab(Image & IMG, image_t & im)
{
  /********** Convert image to MATLAB style representation **********/
  im.N1 = IMG.getHeight();
  im.N2 = IMG.getWidth();
  im.K  = IMG.getPixelSize();
  im.I = (float *) calloc(im.N1*im.N2*im.K, sizeof(float));
  for(int k = 0; k < im.K; k++)
    for(int col = 0; col < im.N2; col++)
      for(int row = 0; row < im.N1; row++)
      {
        unsigned char * pt = IMG.getPixelPt(col, im.N1-1-row);
        im.I[row + col*im.N1 + k*im.N1*im.N2] = 32. * pt[k] / 255.; // Scale 0-32
      }
}

int main(int argc, char ** argv)
{
  //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
  float sigma = 6, tau = 10;
  char * file = "flowers2.pnm";
  char * mode = "gpu";
  char * outover = NULL;
  char * tstr; float tmp;
  if(cutGetCmdLineArgumentstr(argc, (const char**) argv, "file", &tstr))
    file = tstr;
  if(cutGetCmdLineArgumentstr(argc, (const char**) argv, "mode", &tstr))
    mode = tstr;
  if(cutGetCmdLineArgumentstr(argc, (const char**) argv, "outfile", &tstr))
    outover = tstr;

  if(cutGetCmdLineArgumentf(argc, (const char**) argv, "tau", &tmp))
    tau = tmp;
  if(cutGetCmdLineArgumentf(argc, (const char**) argv, "sigma", &tmp))
    sigma = tmp;

  if(cutCheckCmdLineFlag(argc, (const char **)argv, "device"))
    cutilDeviceInit(argc, argv);
  else
    cudaSetDevice(cutGetMaxGflopsDeviceId());

  char * modes[1];
  modes[0] = mode;
  int nmodes = 1;


  /********** Read image **********/
  Image IMG;
  char outfile[1024];

  std::ifstream ifs(file, std::ios::binary);
  if (!ifs) {
      throw Exception("Could not open the file");
  }
  ifs>>IMG;
  image_t im;

  image_to_matlab(IMG, im);

  unsigned int totaltimer;
  cutilCheckError( cutCreateTimer(&totaltimer) );
  cutilCheckError( cutResetTimer(totaltimer) );
  cutilCheckError( cutStartTimer(totaltimer) );

  /********** CUDA setup **********/
  unsigned int timer;

  cutilCheckError( cutCreateTimer(&timer) );

  float *map, *E, *gaps;
  int * flatmap;
  image_t imout;

  map          = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
  gaps         = (float *) calloc(im.N1*im.N2, sizeof(float)) ;
  E            = (float *) calloc(im.N1*im.N2, sizeof(float)) ;

  for(int m = 0; m < nmodes; m++)
  {
    cutilCheckError( cutResetTimer(timer) );
    cutilCheckError( cutStartTimer(timer) );

    /********** Quick shift **********/
    printf("Mode: %s\n", modes[m]);
    if(!strcmp(modes[m], "cpu"))
      quickshift(im, sigma, tau, map, gaps, E);
    else if(!strcmp(modes[m], "gpu"))
      quickshift_gpu(im, sigma, tau, map, gaps, E);
    else
      assert(0 && "Unrecognized mode line");

    cutilCheckError( cutStopTimer(timer) );
    float modeTime = cutGetTimerValue(timer);

    /* Consistency check */
    for(int p = 0; p < im.N1*im.N2; p++)
      if(map[p] == p) assert(gaps[p] == INF);

    flatmap = map_to_flatmap(map, im.N1*im.N2);
    imout = imseg(im, flatmap);
    
    sprintf(outfile, "%s", file);
    char * c = strrchr(outfile, '.');
    if(c) *c = '\0';
    sprintf(outfile, "%s-%s.pnm", outfile, modes[m]); 

    if(outover)
      write_image(imout, outover);
    else
      write_image(imout, outfile);

    free(flatmap);
    free(imout.I);

    printf("Time: %fms\n\n\n", modeTime);
  }

  cutilCheckError( cutStopTimer(totaltimer) );
  float totalTime = cutGetTimerValue(totaltimer);
  printf("Total time: %fms\n", totalTime);


  /********** Cleanup **********/
  free(im.I);

  free(map);
  free(E);
  free(gaps);
}
