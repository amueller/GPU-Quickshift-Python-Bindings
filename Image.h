/* file:        Image.h
** author:      Andrea Vedaldi
** description: Declaration of class Image.
*/
#ifndef _IMAGE_H_
#define _IMAGE_H_

#include"Exception.h"

#include<iostream>

extern "C" {
#include<assert.h>
}

/* --------------------------------------------------------------------
 *                                                       ImageException
 * ----------------------------------------------------------------- */

class ImageException : public Exception 
{ 
public:
  ImageException(const std::string& msg) ;
} ;

inline 
ImageException::ImageException(const std::string& msg) 
  : Exception(msg) 
{ }  

/* --------------------------------------------------------------------
 *                                                                Image
 * ----------------------------------------------------------------- */

class Image
{
public:
  enum PixelType { L, RGB } ;

  Image() ;
  Image(const Image& I) ;
  Image(PixelType type, int w, int h) ;
  ~Image() ;
  Image& operator = (const Image& I)  ;

  unsigned char* getDataPt() ;
  const unsigned char* getDataPt() const ;
  int getDataSize() const ;
  int getWidth() const ;
  int getHeight() const ;
  PixelType getPixelType() const ;
  int getPixelSize() const ;
  unsigned char* getPixelPt(int x, int y) ;
  const unsigned char* getPixelPt(int x, int y) const ;

  std::ostream& putDebug(std::ostream& os) const ;

private:
  PixelType type ;
  unsigned char* pt ;
  int width ;
  int height ;

  friend std::istream& operator>>(std::istream& is, Image& I) ; 
} ;

std::ostream& operator<<(std::ostream& os, Image::PixelType type) ;
std::ostream& operator<<(std::ostream& os, const Image& I) ;
std::istream& operator>>(std::istream& is, Image& I) ; 

/* --------------------------------------------------------------------
 *                                                       Inline members
 * ----------------------------------------------------------------- */

inline 
unsigned char* 
Image::getDataPt() 
{ 
  return pt ; 
}

inline 
const unsigned char* Image::getDataPt() const 
{ 
  return pt ; 
}

inline 
int Image::getDataSize() const 
{ 
  return getPixelSize()*width*height; 
}

inline 
int Image::getWidth() const 
{
  return width ; 
}

inline 
int Image::getHeight() const 
{ 
  return height ; 
}

inline 
Image::PixelType
Image::getPixelType() const 
{
  return type ; 
}

inline 
int Image::getPixelSize() const
{
  switch(type) {
  case L :
    return 1 ;

  case RGB :
    return 3 ;
    
  default:
    assert(0) ;
  }
  return 0 ;
}

inline 
unsigned char* 
Image::getPixelPt(int x, int y)
{ 
  return  pt + getPixelSize() * (y * width + x) ; 
}

inline 
const unsigned char* 
Image::getPixelPt(int x, int y) const 
{ 
  return  pt + getPixelSize() * (y * width + x) ;
}
 

#endif 
// _IMAGE_H_
