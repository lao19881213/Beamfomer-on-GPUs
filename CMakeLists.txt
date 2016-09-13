# $Id: CMakeLists.txt 27477 2013-11-21 13:08:20Z loose $

lofar_add_package(Interface) # Interfaces between RTCP product components 
lofar_add_package(FCNP)      # BlueGene: Fast Collective Network Protocol
lofar_add_package(CNProc)    # BlueGene: Compute-Node Processing applications 
lofar_add_package(IONProc)   # BlueGene: I/O Node Processing applications 
lofar_add_package(Storage)   # Stores correlator output in MeasurmentSets
#lofar_add_package(RTCPTools) # Several RTCP tools
lofar_add_package(Run)       # Run scripts to start RTCP components
lofar_add_package(MetaDataGatherer)# Pulls in meta data from OTDB etc
lofar_add_package(Cobalt)    # GPU replacement for the BG/P
