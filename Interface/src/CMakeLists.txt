# $Id: CMakeLists.txt 22857 2012-11-20 08:58:52Z mol $

include(LofarPackageVersion)

set(interface_LIB_SRCS
  Package__Version.cc
  Allocator.cc
  BeamCoordinates.cc
  CN_Mapping.cc
  DataFactory.cc
  FinalMetaData.cc
  Stream.cc
  Parset.cc
  RSPTimeStamp.cc)

# Maybe we shoud use LOFAR_COMPILER_SUITE, because ASM-BGP_COMPILER_WORKS
# is not reliable: i.e.
#if(LOFAR_COMPILER_SUITE STREQUAL BGPCN)
if(CMAKE_ASM-BGP_COMPILER_WORKS)
  list(APPEND interface_LIB_SRCS
    BGPAsm.S)
endif(CMAKE_ASM-BGP_COMPILER_WORKS)
#endif(LOFAR_COMPILER_SUITE STREQUAL BGPCN)

lofar_add_library(interface ${interface_LIB_SRCS})

lofar_add_bin_program(versioninterface versioninterface.cc)

