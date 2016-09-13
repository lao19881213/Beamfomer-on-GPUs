# $Id: CMakeLists.txt 27103 2013-10-27 09:51:53Z mol $

include(LofarCTest)

lofar_add_test(tBestEffortQueue tBestEffortQueue.cc)
lofar_add_test(tCorrelatedData tCorrelatedData.cc)
lofar_add_test(tMultiDimArray tMultiDimArray.cc)
lofar_add_test(tSparseSet tSparseSet.cc)

if(UNITTEST++_FOUND)
  lofar_add_test(tParset tParset.cc)
  lofar_add_test(tTABTranspose tTABTranspose.cc)
endif(UNITTEST++_FOUND)
