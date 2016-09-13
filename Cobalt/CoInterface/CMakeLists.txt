# $Id: CMakeLists.txt 27103 2013-10-27 09:51:53Z mol $

lofar_package(CoInterface 1.0 DEPENDS ApplCommon Common Stream)

include(LofarFindPackage)
lofar_find_package(Boost REQUIRED COMPONENTS date_time)
lofar_find_package(UnitTest++)
lofar_find_package(OpenMP REQUIRED)
lofar_find_package(Valgrind)

if(USE_VALGRIND)
  add_definitions(-DUSE_VALGRIND)
endif(USE_VALGRIND)  

add_subdirectory(src)
add_subdirectory(test)
