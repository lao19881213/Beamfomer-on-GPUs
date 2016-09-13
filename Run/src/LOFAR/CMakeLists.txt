# $Id: CMakeLists.txt 14273 2009-10-16 10:08:29Z loose $

install(PROGRAMS 
  __init__.py
  CommandClient.py
  Logger.py
  LogValidators.py
  ObservationID.py
  Parset.py
  ParsetTester.py
  Partitions.py
  RingCoordinates.py
  Stations.py
  DESTINATION bin/LOFAR)
