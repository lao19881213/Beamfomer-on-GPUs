# $Id: CMakeLists.txt 24849 2013-05-08 14:51:06Z amesfoort $

install(FILES
  Correlator.cl
  DelayAndBandPass.cl
  #fft2.cl
  FFT.cl
  FIR.cl
  math.cl
  NewCorrelator.cl
  DESTINATION share/gpu/kernels)
