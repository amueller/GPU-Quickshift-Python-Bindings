/* file:        Exception.h
** author:      Andrea Vedaldi
** description: Declaration of class Exception.
**/

#ifndef _EXCEPTION_H_
#define _EXCEPTION_H_

#include<iostream>
#include<string>

class Exception
{ 
 public:
  Exception() ;
  Exception(const std::string& _msg) ;
  std::string getMessage() const ;
 private:
  std::string msg  ;
} ;

/* --------------------------------------------------------------------
 *                                     Inline methods and documentation
 * ----------------------------------------------------------------- */

/** @class Exception
 **
 ** @brief Generic exception.
 **/

/** @brief Constructs an exception whit an empty message.
 **/
inline 
Exception::Exception() 
  : msg() 
{ }

/** @brief Constructs an exception with the specified message.
 **
 ** @param _msg the exception message.
 **/
inline
Exception::Exception(const std::string& _msg) 
  : msg(_msg) 
{ }

/** @brief Returns the exception message.
 **
 ** @return exception message.
 **/
inline
std::string 
Exception::getMessage() const 
{ return msg ; }

/** @brief Puts the exception message on a stream.
 **
 ** @param os output stream.
 ** @param E excepiton.
 ** @return the stream @a os.
 **/
inline 
std::ostream& 
operator<<(std::ostream& os, const Exception& E) 
{
  return os<<E.getMessage() ;
}

#endif
// _EXCEPTION_H_


