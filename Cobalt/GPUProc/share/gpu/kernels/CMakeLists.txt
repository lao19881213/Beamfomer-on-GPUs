# $Id: CMakeLists.txt 25978 2013-08-08 03:26:50Z amesfoort $

# Detect list of kernel sources to install.
# CMake cannot detect changes in this list, but for kernel sources it doesn't need to.
file(GLOB _kernel_path_sources "*.cu" "*.cuh" "*.cl")
set(_kernel_sources )
foreach(path ${_kernel_path_sources})
  get_filename_component(filename "${path}" NAME)  # basename() (optional)
  list(APPEND _kernel_sources "${filename}")
endforeach()

install(FILES ${_kernel_sources} DESTINATION share/gpu/kernels)

