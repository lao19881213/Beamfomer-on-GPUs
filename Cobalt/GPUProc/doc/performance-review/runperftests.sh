#!/bin/sh

# Make sure PATH and LD_LIBRARY_PATH are set correctly. (and LOFARROOT)

#LOFARROOT=~/root rtcp parsets/bfcs-48st-1sb-16b-int4-16ch-127tabs.parset -p > outerr.log 2>&1


rtcp=rtcp
outdir=out

mkdir -p $outdir

# beamformer
for parset in parsets/bf*.parset
do
  out=$outdir/`basename $parset .parset`.out
  echo -e "Beamformer test 'rtcp $parset -p'\n" > $out

  uname -a >> $out
  date --utc >> $out
  echo >> $out

  $rtcp $parset -p >> $out 2>&1
done

# correlator
for parset in parsets/corr*.parset
do
  out=$outdir/`basename $parset .parset`.out
  echo -e "Correlator test 'rtcp $parset -p'\n" > $out

  uname -a >> $out
  date --utc >> $out
  echo >> $out

  $rtcp $parset -p >> $out 2>&1
done

