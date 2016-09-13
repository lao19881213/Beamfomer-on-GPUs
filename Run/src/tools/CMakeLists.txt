# $Id: CMakeLists.txt 14273 2009-10-16 10:08:29Z loose $

lofar_add_bin_program(packetanalysis packetanalysis.c)
lofar_add_bin_program(analyzer common.c analyzer.c)

if(USE_THREADS)
  lofar_add_bin_program(udp-copy common.c udp-copy.c)
  lofar_add_bin_program(generator common.c generator.c)
endif(USE_THREADS)

