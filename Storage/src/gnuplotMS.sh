#!/bin/sh
DATA=`mktemp`
if ! `dirname $0`/plotMS $@ >$DATA
then
  cat $DATA
  rm $DATA
  exit
fi  

OBS=`basename $PWD`
INFO=`<$DATA perl -ne 'print $1 if /# baseline (.*)/;' 2>/dev/null`
echo '
set terminal dumb 120 30
set key outside right
set logscale y
set title "'$OBS'\\nbaseline '$INFO'\\n"
set ylabel "power"
set xlabel "time (s)"
plot "'$DATA'" u 1:2 pt 15 t "X-X", "" u 1:3 pt 24 t "X-Y", "" u 1:4 pt 24 t "Y-X", "" u 1:5 pt 15 t "Y-Y"
' | gnuplot

rm $DATA
