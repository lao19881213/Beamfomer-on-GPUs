# $Id: CMakeLists.txt 21346 2012-06-28 10:10:05Z mol $

include(LofarPackageVersion)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

# Add current source directory to -I path. This is needed because CNProc uses
# angle brackets for internal header files, instead of quotes.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(cnproc_LIB_SRCS
  Package__Version.cc
  AsyncCommunication.cc
  AsyncTranspose.cc
  AsyncTransposeBeams.cc
  BandPass.cc
  BeamFormer.cc
  CN_Processing.cc
  Correlator.cc
  Dedispersion.cc
  FCNP_ClientStream.cc
  FIR.cc
  FilterBank.cc
  InversePPF.cc
  LocationInfo.cc
  PPF.cc
  Flagger.cc
  PreCorrelationFlagger.cc
  PreCorrelationNoChannelsFlagger.cc
  PostCorrelationFlagger.cc
  Stokes.cc)

# Maybe we shoud use LOFAR_COMPILER_SUITE, because ASM-BGP_COMPILER_WORKS
# is not reliable: i.e.
#if(LOFAR_COMPILER_SUITE STREQUAL BGPCN)
if(CMAKE_ASM-BGP_COMPILER_WORKS)
  list(APPEND cnproc_LIB_SRCS
    BeamFormerAsm.S
    ContainsOnlyZerosAsm.S
    CorrelatorAsm.S
    DedispersionAsm.S
    FIR_Asm.S
    FFT_Asm.S
    StokesAsm.S)
endif(CMAKE_ASM-BGP_COMPILER_WORKS)
#endif(LOFAR_COMPILER_SUITE STREQUAL BGPCN)

lofar_add_library(cnproc ${cnproc_LIB_SRCS})

lofar_add_bin_program(CN_Processing CN_Processing_main.cc)
lofar_add_bin_program(versioncnproc versioncnproc.cc)

