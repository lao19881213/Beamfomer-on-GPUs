#!/bin/sh
# apply_casacore_measures_tables.sh
# Install downloaded casacore measures tables atomically and verify which tables are in use.
# Written for jenkins@fs5 (DAS-4).
#
# Known bug:
#  * This script does not work if findmeastable cannot be located or if
#    findmeastable finds the Observatory tables in a directory different
#    from that set locally in $working_dir
#   
# $Id: apply_casacore_measures_data.sh 27117 2013-10-28 20:04:03Z amesfoort $

# Keep these vars in sync with get_casacore_measures_tables.sh
working_dir=$HOME/root/share/casacore  
dir_prefix=IERS-


# find the latest
if ! cd "$working_dir"; then exit 1; fi
update_id=`ls -d $dir_prefix* 2> /dev/null | tail -n 1`
if [ -z "$update_id" ]; then
  echo "No casacore measures directory recognized. Running findmeastable to see if it has it elsewhere."
  findmeastable
  exit
fi

# If not already in use, switch data/ symlink to the latest data/ _atomically_ with a rename. Avoids race with a reader.
if [ "`readlink data`" != "$update_id/data" ]; then
  echo "Applying $update_id"
  ln -s "$update_id/data" "data_${update_id}_tmp" && mv -Tf "data_${update_id}_tmp" data
else
  echo "No new table to apply."
fi

# See if casacore uses the latest tables by extracting the path (token(s) 6,...) from findmeastable.
# If ok, findmeastable prints: "Measures table Observatories found as /home/jenkins/root/share/casacore/data/geodetic/Observatories"
if ! findmeastable > /dev/null; then exit 1; fi
used_dir=`findmeastable | cut -d' ' -f 6-`
if [ $? -ne 0 ]; then exit 1; fi
used_path=`readlink -f "$used_dir/../../../data"`
if [ $? -ne 0 ]; then exit 1; fi
update_id_path="$working_dir/$update_id/data"
if [ "$update_id_path" != "$used_path" ]; then
  echo "It appears that the most recently retrieved measures data is not in use. Most recent is: '$update_id/data'."
  # potential improvement: revert if applied (and if it used to work) (e.g. empty $update_id/)
  exit 1
fi
echo "All cool. The most recently retrieved measures data is (now) in use."

