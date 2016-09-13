# $Id: CMakeLists.txt 27265 2013-11-06 13:28:18Z klijn $

include(LofarCTest)

# Add path to our CUDA kernels to the -I path for NVCC
cuda_include_directories(${PACKAGE_SOURCE_DIR}/share/gpu/kernels)


if(UNITTEST++_FOUND)
  # CUDA CPU side UnitTest++ tests
  lofar_add_test(tGPUWrapper tGPUWrapper.cc)
  lofar_add_test(tCoherentStokes tCoherentStokes.cc)
  lofar_add_test(tIncoherentStokes tIncoherentStokes.cc)
  # GPU kernel UnitTest++ tests
  lofar_add_test(tDelayAndBandPass tDelayAndBandPass.cc)
  lofar_add_test(tBandPassCorrection tBandPassCorrection.cc)
endif(UNITTEST++_FOUND)

# CUDA CPU side unit tests
lofar_add_test(tKernel tKernel.cc)
lofar_add_test(tStreamReadBuffer tStreamReadBuffer.cc)
lofar_add_test(tMultiDimArrayHostBuffer tMultiDimArrayHostBuffer.cc)

# GPU kernel unit tests
lofar_add_test(tIntToFloat tIntToFloat.cc)
lofar_add_test(tFIR_Filter tFIR_Filter.cc)
lofar_add_test(tFFT tFFT.cc)
#lofar_add_executable(tFFT_leakage tFFT_leakage.cc)  # not an automatic test, keep buildable (doesn't right now)
lofar_add_test(tCorrelator tCorrelator.cc)
lofar_add_test(tBeamFormer tBeamFormer.cc)
lofar_add_test(tTranspose tTranspose.cc)
lofar_add_test(tTranspose2 tTranspose2.cc)
lofar_add_test(tCoherentStokesTranspose tCoherentStokesTranspose.cc)
lofar_add_test(tIncoherentStokesTranspose tIncoherentStokesTranspose.cc)

# Trick tests that compile kernels that need to be installed into thinking
# that they can find the kernels using the LOFARROOT environment variable.
set_tests_properties(
  tFIR_Filter
  tCorrelator
  tIntToFloat
  tBeamFormer
  tTranspose
  tTranspose2
  tIncoherentStokesTranspose
  tCoherentStokesTranspose
  PROPERTIES ENVIRONMENT "LOFARROOT=${PACKAGE_SOURCE_DIR}"
)

if(UNITTEST++_FOUND)
  set_tests_properties(
    tDelayAndBandPass
    tBandPassCorrection
    tCoherentStokes
    tIncoherentStokes
    PROPERTIES ENVIRONMENT "LOFARROOT=${PACKAGE_SOURCE_DIR}"
  )
endif(UNITTEST++_FOUND)


# Tests built by nvcc

# To compile .cu files, invoke CUDA's nvcc, but respect the host C/C++ compiler (variants) selection.
set(COMPILER_BINDIR_OPT --compiler-bindir ${CMAKE_CXX_COMPILER})

# The t_cuda_complex test has been disabled, because we no longer need complex.cuh (deleted),
# but do leave the cmake code to compile .cu files, which we probably will need again.
#cuda_add_executable(t_cuda_complex t_cuda_complex.cu OPTIONS ${COMPILER_BINDIR_OPT})
#add_test(t_cuda_complex ${CMAKE_CURRENT_SOURCE_DIR}/t_cuda_complex.sh)


