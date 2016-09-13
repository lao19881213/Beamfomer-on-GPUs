# $Id: CMakeLists.txt 25957 2013-08-06 17:49:54Z amesfoort $

include(LofarCTest)

# Add current directory to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

lofar_add_test(tContext tContext.cc)
lofar_add_test(tPerformanceCounter tPerformanceCounter.cc)
lofar_add_test(RTCP_UnitTest RTCP_UnitTest.cc UnitTest.cc)

