# $Id: CMakeLists.txt 24442 2013-03-28 09:09:18Z loose $

include(LofarPackageVersion)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

lofar_add_library(outputproc 
  Package__Version.cc
  ExitOnClosedStdin.cc
  FastFileStream.cc
  InputThread.cc
  OutputThread.cc
  SubbandWriter.cc
  MSWriter.cc
  MSWriterCorrelated.cc
  MSWriterDAL.cc
  MSWriterNull.cc
  MSWriterFile.cc
  Format.cc
  MeasurementSetFormat.cc
  TBB_StaticMapping.cc
#  TBB_Writer.cc
)

install(PROGRAMS
  gnuplotMS.sh
  DESTINATION bin)

install(FILES
  outputProc.log_prop
  DESTINATION etc)

lofar_add_bin_program(outputProc outputProc.cc)
#lofar_add_bin_program(createHeaders createHeaders.cc)
lofar_add_bin_program(plotMS plotMS.cc)
lofar_add_bin_program(versionoutputproc versionoutputproc.cc)
#lofar_add_bin_program(TBB_Writer_main TBB_Writer_main.cc)

