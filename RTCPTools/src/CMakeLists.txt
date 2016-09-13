# $Id: CMakeLists.txt 24191 2013-03-11 15:42:54Z amesfoort $

find_library(M_LIBRARY m)
find_library()

add_executable(generator generator.c)
target_link_libraries(generator ${M_LIBRARY})

add_executable(udp-copy udp-copy.c)

add_executable(tbb-crc-test tbb-crc-test.cc)
add_executable(tbb-dumpframes tbb-dumpframes.cc)

