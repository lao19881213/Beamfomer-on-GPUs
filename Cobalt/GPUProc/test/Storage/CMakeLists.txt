# $Id: CMakeLists.txt 25295 2013-06-12 11:18:33Z loose $

include(LofarCTest)

# DummyStorage is started by tStorageProcesses to emulate an OutputProc process
lofar_add_executable(DummyStorage DummyStorage.cc)
lofar_add_test(tStorageProcesses tStorageProcesses.cc DEPENDS DummyStorage)

