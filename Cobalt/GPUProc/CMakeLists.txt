# $Id: CMakeLists.txt 26639 2013-09-23 15:58:03Z amesfoort $

# Handle options USE_CUDA and USE_OPENCL.
if(USE_CUDA AND NOT USE_OPENCL)
  # We need at least CMake 2.8 for CUDA support
  cmake_minimum_required(VERSION 2.8)

  set(_gpuproc_deps Common Stream ApplCommon CoInterface InputProc)
  lofar_find_package(CUDA 4.1 REQUIRED)
  lofar_find_package(CUDADriver REQUIRED)

  add_definitions(-DUSE_CUDA)
  if(LOFAR_BUILD_VARIANT MATCHES "^DEBUG$")
    list(APPEND CUDA_NVCC_FLAGS -g -G)
  endif()

  if(0)  # for dev to see what is available how; disabled to reduce spam
    message(STATUS "CUDA_FOUND = ${CUDA_FOUND}")
    message(STATUS "CUDA_VERSION_MAJOR = ${CUDA_VERSION_MAJOR}")
    message(STATUS "CUDA_VERSION_MINOR = ${CUDA_VERSION_MINOR}")
    message(STATUS "CUDA_VERSION = ${CUDA_VERSION}") # sometimes avail if CUDA_VERSION_STRING is not
    message(STATUS "CUDA_VERSION_STRING = ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_SDK_ROOT_DIR = ${CUDA_SDK_ROOT_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_CUFFT_LIBRARIES = ${CUDA_CUFFT_LIBRARIES}")
    message(STATUS "CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}")
    message(STATUS "CUDA_curand_LIBRARY = ${CUDA_curand_LIBRARY}")
    message(STATUS "CUDA_cusparse_LIBRARY = ${CUDA_cusparse_LIBRARY}")
    message(STATUS "CUDA_npp_LIBRARY = ${CUDA_npp_LIBRARY}")
    message(STATUS "CUDA_nvcuvenc_LIBRARY = ${CUDA_nvcuvenc_LIBRARY}")
    message(STATUS "CUDA_nvcuvid_LIBRARY = ${CUDA_nvcuvid_LIBRARY}")

    message(STATUS "CUDADRIVER_FOUND = ${CUDADRIVER_FOUND}")
    message(STATUS "CUDADRIVER_LIBRARIES = ${CUDADRIVER_LIBRARIES}")
  endif()

elseif(USE_OPENCL AND NOT USE_CUDA)
  set(_gpuproc_deps Common Stream ApplCommon CoInterface InputProc OpenCL_FFT)
  lofar_find_package(OpenCL REQUIRED)
  add_definitions(-DUSE_OPENCL)
else()
  message(FATAL_ERROR
    "Either CUDA or OpenCL must be enabled to build GPUProc.")
endif()

lofar_package(GPUProc 1.0 DEPENDS ${_gpuproc_deps})

lofar_find_package(OpenMP REQUIRED)
lofar_find_package(Boost REQUIRED)
lofar_find_package(MPI)
lofar_find_package(LibNuma)
lofar_find_package(FFTW3 COMPONENTS single double threads REQUIRED)  # 'double threads' for FFT unit test refs
lofar_find_package(UnitTest++)

# LibSSH2 can be linked against either OpenSSL or GCrypt. We need to
# find the right one, and initialise it for multi-threaded execution.
lofar_find_package(LibSSH2 REQUIRED)
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/links_gcrypt.sh ${LIBSSH2_LIBRARY}
  RESULT_VARIABLE LIBSSH2_LINKS_GCRYPT)

if (LIBSSH2_LINKS_GCRYPT MATCHES 0)
  lofar_find_package(GCrypt REQUIRED)
else()
  lofar_find_package(OpenSSL REQUIRED)
endif()

# InputProc/Delays.h drags in a dependency on casa; hence we need to add the
# following line. This should be fixed someday.
lofar_find_package(Casacore COMPONENTS casa)

if(USE_VALGRIND)
  lofar_find_package(Valgrind)
  if(VALGRIND_FOUND)
    add_definitions(-DUSE_VALGRIND)
  else()
    message(WARNING "USE_VALGRIND requested, but not found!")
  endif()
endif()  

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(share/gpu/kernels)
add_subdirectory(etc)
