# $Id: CMakeLists.txt 24124 2013-03-08 10:12:42Z mol $

include(LofarPackageVersion)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

# Add current source directory to -I path. This is needed because IONProc uses
# angle brackets for internal header files, instead of quotes.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

lofar_add_library(ionproc
  Package__Version.cc
  BeamletBuffer.cc
  BeamletBufferToComputeNode.cc
  CommandServer.cc
  ControlPhase3Cores.cc
  Delays.cc
  GlobalVars.cc
  InputSection.cc
  InputThread.cc
  InputThreadAsm.S
  ION_Allocator.cc
  Job.cc
  JobQueue.cc
  LogThread.cc
  OutputThread.cc
  OutputSection.cc
  ReaderWriterSynchronization.cc
  Scheduling.cc
  StorageProcesses.cc
  StreamMultiplexer.cc
  SSH.cc
  FCNP_ServerStream.cc)

install(PROGRAMS
  recreateFeedback.sh
  DESTINATION bin)

lofar_add_bin_program(versionionproc versionionproc.cc)
lofar_add_bin_program(ION_Processing ION_main.cc)
lofar_add_bin_program(generateDelays generateDelays.cc)
