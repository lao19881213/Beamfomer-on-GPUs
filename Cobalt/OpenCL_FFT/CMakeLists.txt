# $Id: CMakeLists.txt 24987 2013-05-21 16:54:28Z amesfoort $

lofar_package(OpenCL_FFT 1.0)

if(USE_OPENCL)
  include(LofarFindPackage)
  lofar_find_package(OpenCL REQUIRED)
  add_subdirectory(src)
endif()
