# $Id: CMakeLists.txt 26385 2013-09-05 15:00:52Z schoenmakers $

lofar_package(CNProc 1.0 DEPENDS Common Stream Interface FCNP)

add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)

include(LofarFindPackage)
lofar_find_package(MASS)
lofar_find_package(Boost REQUIRED)
lofar_find_package(FFTW3 COMPONENTS single)
if(NOT FFTW3_FOUND)
  lofar_find_package(FFTW2 COMPONENTS single real)
  if(NOT FFTW2_FOUND)
    message(SEND_ERROR "Should have FFTW3 or FFTW2 installed.")
  endif(NOT FFTW2_FOUND)
endif(NOT FFTW3_FOUND)

lofar_find_package(Valgrind)
if(USE_VALGRIND)
  add_definitions(-DUSE_VALGRIND)
endif(USE_VALGRIND)  

## ---------------------------------------------------------------------------
## Enable BGP specific assembler.
## Use the BGP assembler also for linking C/C++ programs.
##
## NOTE: Maybe this should be moved to the variants file. That requires that
## CN and ION are no longer variants; we then need two compiler definitions:
## BGPCN and BGPION, which is a cleaner solution anyway.
## ---------------------------------------------------------------------------
enable_language(ASM-BGP)
if(LOFAR_COMPILER_SUITE STREQUAL "BGPCN")
  set(CMAKE_C_LINK_EXECUTABLE ${CMAKE_ASM-BGP_LINK_EXECUTABLE})
  set(CMAKE_CXX_LINK_EXECUTABLE ${CMAKE_ASM-BGP_LINK_EXECUTABLE})
endif(LOFAR_COMPILER_SUITE STREQUAL "BGPCN")

add_subdirectory(src)
add_subdirectory(test)
