# $Id: CMakeLists.txt 26385 2013-09-05 15:00:52Z schoenmakers $

lofar_package(RTCPTools 1.0)

include(LofarFindPackage)
lofar_find_package(Boost REQUIRED)

add_subdirectory(src)
