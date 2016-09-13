# List of header files that will be installed.
set(inst_HEADERS
  Align.h
  AlignedStdAllocator.h
  Allocator.h
  BeamCoordinates.h
  BeamFormedData.h
  BGPAsm.h
  CN_Command.h
  CN_Mapping.h
  Stream.h
  CorrelatedData.h
  Config.h
  FilteredData.h
  FakeData.h
  MultiDimArray.h
  Parset.h
  SubbandMetaData.h
  PrintVector.h
  RSPTimeStamp.h
  SparseSet.h
  Exceptions.h
  StreamableData.h
  TransposedData.h
  TriggerData.h
  InverseFilteredData.h
  InputData.h)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

# Install header files.
install(FILES ${inst_HEADERS} DESTINATION include/${PACKAGE_NAME})
