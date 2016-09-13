# $Id: CMakeLists.txt 26385 2013-09-05 15:00:52Z schoenmakers $

lofar_package(Interface 1.0 DEPENDS ApplCommon Common Stream)

include(LofarFindPackage)
lofar_find_package(Boost REQUIRED COMPONENTS date_time)
lofar_find_package(Valgrind)

if(USE_VALGRIND)
  add_definitions(-DUSE_VALGRIND)
endif(USE_VALGRIND)  

enable_language(ASM-BGP)

add_subdirectory(include/Interface)
add_subdirectory(src)
add_subdirectory(test)
