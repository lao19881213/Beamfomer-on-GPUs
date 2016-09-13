# $Id: CMakeLists.txt 19129 2011-10-31 08:54:08Z mol $

include(LofarCTest)

include_directories(${PACKAGE_SOURCE_DIR}/src) 

lofar_add_test(tCN_Processing tCN_Processing.cc)
lofar_add_test(tCN_Processing_alloc tCN_Processing_alloc.cc)
lofar_add_test(tBeamForming tBeamForming.cc)
lofar_add_test(tDedispersion tDedispersion.cc)
lofar_add_test(tFIR_Asm tFIR_Asm.cc)
lofar_add_test(tPencilBeamFormer tPencilBeamFormer.cc)
lofar_add_test(tStokes tStokes.cc)
lofar_add_test(tInversePPF tInversePPF.cc)
lofar_add_test(tStokesAsm tStokesAsm.cc)
