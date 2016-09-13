# $Id: CMakeLists.txt 26385 2013-09-05 15:00:52Z schoenmakers $

lofar_package(Storage 1.0 DEPENDS Common ApplCommon Stream Interface MSLofar LofarStMan)

include(LofarFindPackage)
lofar_find_package(Casacore COMPONENTS casa ms tables REQUIRED)
lofar_find_package(DAL REQUIRED)
lofar_find_package(HDF5 REQUIRED)
lofar_find_package(Boost REQUIRED)

add_subdirectory(include/Storage)
add_subdirectory(src)
add_subdirectory(test)
