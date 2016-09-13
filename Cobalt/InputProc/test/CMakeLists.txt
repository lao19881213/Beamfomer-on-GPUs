# $Id: CMakeLists.txt 26881 2013-10-07 07:45:48Z loose $

include(LofarCTest)

lofar_add_test(tSharedMemory tSharedMemory.cc)
lofar_add_test(tSampleBuffer tSampleBuffer.cc)
if(UNITTEST++_FOUND)
  lofar_add_test(tRanges tRanges.cc)
  lofar_add_test(tSampleBufferSync tSampleBufferSync.cc)

  lofar_add_test(tDelays tDelays.cc)
  lofar_add_test(tRSPTimeStamp tRSPTimeStamp.cc)
endif(UNITTEST++_FOUND)

lofar_add_test(tRSP tRSP.cc)
lofar_add_test(tPacketReader tPacketReader.cc)
lofar_add_test(tPacketFactory tPacketFactory.cc)
lofar_add_test(tGenerator tGenerator.cc)
lofar_add_test(t_generateRSP t_generateRSP.cc)

lofar_add_test(tPacketWriter tPacketWriter.cc)
lofar_add_test(tPacketsToBuffer tPacketsToBuffer.cc)

if(UNITTEST++_FOUND)
  lofar_add_test(tBufferSettings tBufferSettings.cc)
  lofar_add_test(tBlockReader tBlockReader.cc)
  if(MPI_FOUND)
    lofar_add_test(tMPISendReceiveStation tMPISendReceiveStation.cc)
  endif(MPI_FOUND)
endif(UNITTEST++_FOUND)

if(MPI_FOUND)
  lofar_add_test(tMPITransfer tMPITransfer.cc)
  lofar_add_test(tMPI tMPI.cc)
endif(MPI_FOUND)
