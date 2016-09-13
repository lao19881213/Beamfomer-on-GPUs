# $Id: CMakeLists.txt 27477 2013-11-21 13:08:20Z loose $

lofar_add_package(InputProc)   # GPU cluster: Station Input Retrieval and Redistribution applications 
lofar_add_package(OutputProc)  # GPU cluster: Data Storage applications
lofar_add_package(GPUProc)     # GPU computation applications
lofar_add_package(OpenCL_FFT)  # Apple's FFT library for OpenCL
lofar_add_package(CoInterface) # CoInterface code between Cobalt packages
