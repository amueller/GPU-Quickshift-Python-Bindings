cmake_minimum_required (VERSION 2.6)
project (py-quickshift-gpu)

# CPACK options for debian package
set(CPACK_PACKAGE_NAME pyquickshift)
set(CPACK_PACKAGE_VERSION 0.2.1)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
set(CPACK_PACKAGE_CONTACT "amueller@ais.uni-bonn.de")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Fulkersons very fast quick shift with additional python bindings. Please site his paper.")
INCLUDE(CPack)

set(Boost_USE_STATIC_LIBS FALSE)

FIND_PACKAGE( Boost 1.37 COMPONENTS python REQUIRED )
FIND_PACKAGE( PythonLibs )

SET (CUDA_SDK_ROOT_DIR "/usr/local/cuda/C" CACHE STRING "Location of CUDA SDK") 
FIND_PACKAGE(CUDA)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_13")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_10,code=\"sm_10,compute_10\";-gencode=arch=compute_20,code=\"sm_20,compute_20\"")

CUDA_INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/common/inc ${CMAKE_CURRENT_SOURCE_DIR}  tools )
INCLUDE_DIRECTORIES(      ${CUDA_SDK_ROOT_DIR}/common/inc ${CUDA_INCLUDE_DIRS}         tools )

link_directories(${CUDA_SDK_ROOT_DIR}/lib)
cuda_add_library(quickshift_gpu SHARED
    quickshift_gpu.cu
    )
MESSAGE(STATUS "CUDA_SDK: ${CUDA_SDK_ROOT_DIR}")
target_link_libraries(quickshift_gpu
    cutil
)

include_directories( ${PYTHON_INCLUDE_PATH})
PYTHON_ADD_MODULE(pyquickshift SHARED quickshift_python.cpp)
target_link_libraries(pyquickshift
    ${Boost_PYTHON_LIBRARY_RELEASE}
    ${PYTHON_LIBRARY}
    quickshift_gpu
    )

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS pyquickshift  DESTINATION ${PYTHON_SITE_PACKAGES}/quickshift/)
install(FILES __init__.py  DESTINATION ${PYTHON_SITE_PACKAGES}/quickshift/)
install(FILES lena.png  DESTINATION ${PYTHON_SITE_PACKAGES}/quickshift/)
install(TARGETS quickshift_gpu DESTINATION lib)
