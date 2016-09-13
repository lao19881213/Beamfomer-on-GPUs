# $Id: CMakeLists.txt 27103 2013-10-27 09:51:53Z mol $

include(LofarPackageVersion)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

lofar_add_library(cointerface
  Package__Version.cc
  Allocator.cc
  BeamCoordinates.cc
  BlockID.cc
  DataFactory.cc
  FinalMetaData.cc
  Stream.cc
  Parset.cc
  TABTranspose.cc)

lofar_add_bin_program(versioncointerface versioncointerface.cc)

