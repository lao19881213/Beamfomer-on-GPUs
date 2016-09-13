# $Id: CMakeLists.txt 27103 2013-10-27 09:51:53Z mol $

include(LofarCTest)

# Add current source directory to -I path. This is needed because some
# source files and headers in subdirectories include UnitTest.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

lofar_add_test(tSSH tSSH.cc)
lofar_add_test(tfpequals tfpequals.cc)
lofar_add_test(tstartBGL DEPENDS rtcp)
lofar_add_test(tMACfeedback DEPENDS rtcp)
lofar_add_test(tProductionParsets DEPENDS rtcp)

# cmpfloat is started by scripts for a fuzzy compare of 2 output files with raw floats
lofar_add_executable(cmpfloat cmpfloat.cc)

# tests that use testParset.sh

# fail on all system skip for now
lofar_add_test(tBeamform_1sec_1st_5sb_noflagging DEPENDS rtcp)  

lofar_add_test(tCorrelate_1sec_1st_5sb_noflagging DEPENDS rtcp cmpfloat)
lofar_add_test(tCorrelate_3sec_1st_5sb DEPENDS rtcp cmpfloat)
lofar_add_test(tCorrelate_3sec_2st_5sb DEPENDS rtcp cmpfloat)
lofar_add_test(tBeamform_incoherent_2sec_1st_5sb_noflagging DEPENDS rtcp cmpfloat)
#lofar_add_test(tGPUPerformance DEPENDS rtcp) # performance measurement and comparison are broken
set_tests_properties(
  tCorrelate_1sec_1st_5sb_noflagging
  tCorrelate_3sec_1st_5sb
  tCorrelate_3sec_2st_5sb
  tBeamform_1sec_1st_5sb_noflagging
  tBeamform_incoherent_2sec_1st_5sb_noflagging
  PROPERTIES RUN_SERIAL ON
)

if(MPI_FOUND)
  lofar_add_test(t_cpu_utils t_cpu_utils.cc)
  set_tests_properties(
    t_cpu_utils
    PROPERTIES RUN_SERIAL ON
  )
endif()

if(UNITTEST++_FOUND)
  lofar_add_test(t_gpu_utils t_gpu_utils)
  lofar_add_test(tRunningStatistics tRunningStatistics.cc)
endif()

add_subdirectory(Kernels)
add_subdirectory(Pipelines)
add_subdirectory(Storage)
add_subdirectory(SubbandProcs)

if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_OPENCL)
  add_subdirectory(opencl)
endif()
