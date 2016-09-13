#!/bin/bash
#
# Usage: ./generateHardwareMap.sh > CobaltHardwareMap.parset

perl -e '

open $fh, "StationStreams.parset"
  or die "Cannot open StationStreams.parset";

$numhosts = 8;
$numranks = $numhosts * 2;

for $rank (1..$numranks) {
  push @stations, "";
}

while ($line = <$fh>) {
  next unless $line =~ /PIC.Core.Station.([A-Z0-9]+).RSP.ports = \[udp:10.168.([0-9]+).([1234])/;

  $station = $1;
  $hostnr = $2 - 95;
  $ifacenr = $3;

  if ($ifacenr == 1 or $ifacenr == 2) {
    $socket = 0;
  } else {
    $socket = 1;
  }

  $rank = ($hostnr - 1) * 2 + $socket;

  if ($stations[$rank] ne "") {
    $stations[$rank] .= ", ";
  }
  $stations[$rank] .= $station;
}

printf "Cobalt.Hardware.nrNodes=%s\n", $numranks;

for $rank (0 .. $numranks-1) {
  $hostnr = $rank / 2 + 1;
  $socket = $rank % 2;
  $firstgpu = $socket * 2;
  $secondgpu = $firstgpu + 1;

  printf "Cobalt.Hardware.Node[%s].host=cbm%03d\n", $rank, $hostnr;
  printf "Cobalt.Hardware.Node[%s].cpu=%d\n", $rank, $socket;
  printf "Cobalt.Hardware.Node[%s].gpus=[%d, %d]\n", $rank, $firstgpu, $secondgpu;
  printf "Cobalt.Hardware.Node[%s].stations=[%s]\n", $rank, $stations[$rank];
}
'
