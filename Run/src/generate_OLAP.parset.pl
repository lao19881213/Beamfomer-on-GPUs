#!/usr/bin/perl
#
# usage:
#
# ./generate_OLAP.parset *-AntennaField.conf > OLAP.parset
#
# The AntennaField configuration files are in
# MAC/Deployment/data/StaticMetaData/AntennaFields/


while(<>) {
  # "AntennaPositions for CS001" 
  $station = $1 if /AntennaPositions for ([A-Z0-9]+)/;

  # "LBA"
  $field = $1 if /^(LBA|HBA|HBA0|HBA1)$/;

  # empty line moves to next field
  $field = "" if /^\s+$/;
  next if $field eq "";

  # "3 [ 3826923.589520000 460915.393221000 5064643.461 ]"
  /3\s+\[\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+\]/ || next;
  $x = $1;
  $y = $2;
  $z = $3;

  printf "PIC.Core.%s%s.phaseCenter = [%.3f, %.3f, %.3f]\n", $station, $field, $x, $y, $z;
}

