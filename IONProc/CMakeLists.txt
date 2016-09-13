# $Id: CMakeLists.txt 26385 2013-09-05 15:00:52Z schoenmakers $

lofar_package(IONProc 1.0
  DEPENDS Common ApplCommon Stream Interface FCNP)

include(LofarFindPackage)
lofar_find_package(Boost REQUIRED)
lofar_find_package(LibSSH2 REQUIRED)
lofar_find_package(OpenSSL REQUIRED)
lofar_find_package(Valgrind)
lofar_find_package(Casacore COMPONENTS measures REQUIRED)

if(USE_VALGRIND)
  add_definitions(-DUSE_VALGRIND)
endif(USE_VALGRIND)  

enable_language(ASM-BGP)

add_subdirectory(src)
add_subdirectory(test)
