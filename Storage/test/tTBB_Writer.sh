#!/bin/bash

# Test the TBB_Writer.
# Use named pipes (FIFOs) to test multi-threaded writes (not possible with single stream stdin)
# and to have automatic flow-control (unlike with udp-copy).
#
# The TBB validation tool from DAL does some checks on the HDF5 output file. (for the mo, done by a minimal script)
#

touch tTBB_Writer.log

nstreams = 6    # the nr of RSPs in NL stations

parsetfilename = "tTBB_Writer-transient.parset" # does not belong to the test data, but good enough to test
declare -a rawinfilenames
outfilename = "unk.h5"
declare -a rawoutfilenames
for (( i = 0 ; i < $nstreams ; i++ )) ;
do
  ${rawinfilenames[$i]}  = "unkin$i.dat"
  ${rawoutfilenames[$i]} = "unk$i.raw"
  mknod tTBB_Writer_stream$i.pipe p 2>&1 >> tTBB_Writer.log
  cat ${rawinfilenames[$i]} > tTBB_Writer_stream$i.pipe &
done

./runctest.sh TBB_Writer --parsetfile=$parsetfilename --timeout=1 --keeprunning=0

# Ideally, we use h5check first to verify that the file is a proper hdf5 file,
# but h5check is a separate util and usually not installed, so check content straight away.

#can use lofar_tbb_headerinfo.py for a first guess/test
./tTBB_Writer-verify.py $outfilename 2>&1 >> tTBB_Writer.log
STATUS = $?

rm -f $outh5filename 2>&1 >> tTBB_Writer.log
for (( i = 0 ; i < $nstreams ; i++ )) ;
do
  rm -f ${rawoutfilename[$i]} 2>&1 >> tTBB_Writer.log
  rm tTBB_Writer_stream$i.pipe 2>&1 >> tTBB_Writer.log
done

exit $STATUS

