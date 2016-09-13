# $Id: CMakeLists.txt 17003 2011-01-06 08:54:59Z romein $

include(LofarPackageVersion)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

set(OpenCL_FFT_LIB_SRCS
  #Package__Version.cc
  fft_execute.cpp
  fft_kernelstring.cpp
  fft_setup.cpp)

lofar_add_library(opencl_fft ${OpenCL_FFT_LIB_SRCS})

#lofar_add_bin_program(versionopencl_fft versionopencl_fft.cc) #FIXME

