# $Id: CMakeLists.txt 25751 2013-07-24 14:32:26Z mol $

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda_config.h.in"
  "${CMAKE_BINARY_DIR}/include/cuda_config.h"  # internal, no need to install
)

