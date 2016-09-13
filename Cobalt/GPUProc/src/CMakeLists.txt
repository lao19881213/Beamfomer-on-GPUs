# $Id: CMakeLists.txt 27265 2013-11-06 13:28:18Z klijn $

include(LofarPackageVersion)

# Create symbolic link to include directory.
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME})

set(_gpuproc_sources
  #Package__Version.cc
  BandPass.cc
  cpu_utils.cc
  FilterBank.cc
  global_defines.cc
  RunningStatistics.cc
  Station/StationNodeAllocation.cc
  Station/StationInput.cc
  Storage/SSH.cc
  Storage/StorageProcess.cc
  Storage/StorageProcesses.cc
)

if(USE_CUDA)
  list(APPEND _gpuproc_sources 
    cuda/gpu_wrapper.cc
    cuda/gpu_utils.cc
    cuda/KernelFactory.cc
    cuda/PerformanceCounter.cc
    cuda/Kernels/Kernel.cc
    cuda/Kernels/BeamFormerKernel.cc
    cuda/Kernels/BeamFormerTransposeKernel.cc
    cuda/Kernels/CoherentStokesTransposeKernel.cc
    cuda/Kernels/CoherentStokesKernel.cc
    cuda/Kernels/CorrelatorKernel.cc
    #cuda/Kernels/DedispersionBackwardFFTkernel.cc
    #cuda/Kernels/DedispersionChirpKernel.cc
    #cuda/Kernels/DedispersionForwardFFTkernel.cc
    cuda/Kernels/DelayAndBandPassKernel.cc
    cuda/Kernels/BandPassCorrectionKernel.cc
    cuda/Kernels/FFT_Kernel.cc
    cuda/Kernels/FFT_Plan.cc
    cuda/Kernels/Filter_FFT_Kernel.cc
    cuda/Kernels/FIR_FilterKernel.cc
    cuda/Kernels/IncoherentStokesKernel.cc
    cuda/Kernels/IncoherentStokesTransposeKernel.cc
    cuda/Kernels/IntToFloatKernel.cc
    #cuda/Kernels/UHEP_BeamFormerKernel.cc
    #cuda/Kernels/UHEP_InvFFT_Kernel.cc
    #cuda/Kernels/UHEP_InvFIR_Kernel.cc
    #cuda/Kernels/UHEP_TransposeKernel.cc
    #cuda/Kernels/UHEP_TriggerKernel.cc
    cuda/Pipelines/Pipeline.cc
    cuda/Pipelines/BeamFormerPipeline.cc
    cuda/Pipelines/CorrelatorPipeline.cc
#    cuda/Pipelines/UHEP_Pipeline.cc
    cuda/SubbandProcs/SubbandProc.cc
    cuda/SubbandProcs/BeamFormerSubbandProc.cc
    cuda/SubbandProcs/BeamFormerFactories.cc
    cuda/SubbandProcs/CorrelatorSubbandProc.cc
#    cuda/SubbandProcs/UHEP_SubbandProc.cc
  )
  add_subdirectory(cuda)

  lofar_add_library(gpuproc ${_gpuproc_sources})
  if(CUDA_cufft_LIBRARY)
    target_link_libraries(gpuproc ${CUDA_cufft_LIBRARY})
  endif()

  lofar_add_bin_program(rtcp rtcp.cc)
endif()

if(USE_OPENCL)
  list(APPEND _gpuproc_sources 
    opencl/gpu_wrapper.cc
    opencl/gpu_utils.cc
    opencl/PerformanceCounter.cc
    opencl/Kernels/Kernel.cc
    opencl/Kernels/BeamFormerKernel.cc
    opencl/Kernels/BeamFormerTransposeKernel.cc
    opencl/Kernels/CoherentStokesKernel.cc
    opencl/Kernels/CorrelatorKernel.cc
    opencl/Kernels/DedispersionBackwardFFTkernel.cc
    opencl/Kernels/DedispersionChirpKernel.cc
    opencl/Kernels/DedispersionForwardFFTkernel.cc
    opencl/Kernels/DelayAndBandPassKernel.cc
    opencl/Kernels/FFT_Kernel.cc
    opencl/Kernels/FFT_Plan.cc
    opencl/Kernels/Filter_FFT_Kernel.cc
    opencl/Kernels/FIR_FilterKernel.cc
    opencl/Kernels/IncoherentStokesKernel.cc
    opencl/Kernels/IntToFloatKernel.cc
    #opencl/Kernels/UHEP_BeamFormerKernel.cc
    #opencl/Kernels/UHEP_InvFFT_Kernel.cc
    #opencl/Kernels/UHEP_InvFIR_Kernel.cc
    #opencl/Kernels/UHEP_TransposeKernel.cc
    #opencl/Kernels/UHEP_TriggerKernel.cc
    opencl/Pipelines/Pipeline.cc
    opencl/Pipelines/BeamFormerPipeline.cc
    opencl/Pipelines/CorrelatorPipeline.cc
    #opencl/Pipelines/UHEP_Pipeline.cc
    opencl/SubbandProcs/SubbandProc.cc
    opencl/SubbandProcs/BeamFormerSubbandProc.cc
    opencl/SubbandProcs/CorrelatorSubbandProc.cc
    #opencl/SubbandProcs/UHEP_SubbandProc.cc
  )
  add_subdirectory(opencl)

  lofar_add_library(gpuproc_opencl ${_gpuproc_sources})
  #lofar_add_bin_program(rtcp_opencl rtcp.cc)
endif()

lofar_add_bin_program(mpi_node_list Station/mpi_node_list.cc)

# install the scripts for MAC and for auto-updating casacore measures tables
install(PROGRAMS
  scripts/runObservation.sh
  scripts/startBGL.sh
  scripts/stopBGL.sh
  scripts/get_casacore_measures_data.sh
  scripts/apply_casacore_measures_data.sh
  DESTINATION bin)

# install the logprop file
install(FILES
  rtcp.log_prop
  Station/mpi_node_list.log_prop
  DESTINATION etc)
