# $Id: CMakeLists.txt 14273 2009-10-16 10:08:29Z loose $

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/locations.sh.in
  ${CMAKE_CURRENT_BINARY_DIR}/locations.sh
  @ONLY)

install(PROGRAMS 
  routeToCobalt
  startBGL.sh
  stopBGL.sh
  BGPPartition.sh
  CNProcessing.sh
  IONProcessing.sh
  LogArchiver.sh
  controller.sh
  gracefullyStopBGProcessing.sh
  ${CMAKE_CURRENT_BINARY_DIR}/locations.sh
  watchlogs.sh
  commandOLAP.py
  DESTINATION bin)

install(FILES
  BlueGeneControl.conf
  casarc
  multitail-olap.conf
  OLAP.parset
  MAC+IP.dat
  RSPConnections.dat
  DESTINATION etc)

lofar_add_bin_program(CEPlogProcessorSettings CEPlogProcessorSettings.cc)

add_subdirectory(LOFAR)
add_subdirectory(util)
add_subdirectory(tools)
