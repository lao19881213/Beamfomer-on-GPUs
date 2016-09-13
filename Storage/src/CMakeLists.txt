# $Id: CMakeLists.txt 24287 2013-03-15 09:15:13Z mol $

include(LofarPackageVersion)

lofar_add_library(storage 
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

#install(PROGRAMS
#  gnuplotMS.sh
#  DESTINATION bin)

install(FILES
  Storage_main.log_prop
  DESTINATION etc)

lofar_add_bin_program(Storage_main Storage_main.cc)
lofar_add_bin_program(createHeaders createHeaders.cc)
#lofar_add_bin_program(plotMS plotMS.cc)
lofar_add_bin_program(versionstorage versionstorage.cc)
#lofar_add_bin_program(TBB_Writer_main TBB_Writer_main.cc)

