#!/bin/sh
# get_casacore_measures_tables.sh
# Retrieve new casacore measures tables under $working_dir and extract. Written for jenkins@fs5 (DAS-4).
# If it works out, remove very old download dirs starting with $dir_prefix.
#
# $Id: get_casacore_measures_data.sh 27117 2013-10-28 20:04:03Z amesfoort $

# Keep these vars in sync with apply_casacore_measures_tables.sh
working_dir=$HOME/root/share/casacore
dir_prefix=IERS-


update_id=$dir_prefix`date +%FT%T.%N`  # e.g. measures_data-2013-09-26T01:58:30.098006623
if [ $? -ne 0 ]; then exit 1; fi
measures_ftp_path=ftp://ftp.atnf.csiro.au/pub/software/measures_data
measures_data_filename=measures_data.tar.bz2
measures_md5sum_filename=$measures_data_filename.md5sum


# Get the data from CSIRO's (slow from NL) FTP server. About 8 MB may take 30 seconds.
# By default, when wget downloads a file, the timestamp is set to match the timestamp from the remote file.
download()
{
  wget -N --tries=4 \
    $measures_ftp_path/$measures_data_filename \
    $measures_ftp_path/$measures_md5sum_filename
}

# Verify that md5 hash is equal to hash in $measures_md5sum_filename
# No need to compare the filename. (And note that the .md5sum from CSIRO contains a CSIRO path.)
check_md5()
{
  local md5sum=`cut -f 1 -d ' ' $measures_md5sum_filename`
  if [ $? -ne 0 ]; then return 1; fi
  local data_md5=`md5sum $measures_data_filename | cut -f 1 -d ' '`
  if [ -z "$data_md5" ]; then return 1; fi

  if [ "$md5sum" != "$data_md5" ]; then
    echo "Computed and downloaded MD5 sums do not match."
    return 1
  fi

  return 0
}


# Use a tmp_ name until it is ready to avoid racing with apply script.
if ! cd "$working_dir" || ! mkdir "tmp_$update_id" || ! cd "tmp_$update_id"; then exit 1; fi
if ! download || ! check_md5; then
  echo "Download or MD5 checksums check failed. Retrying once."
  rm -f $measures_data_filename $measures_md5sum_filename
  sleep 2
  if ! download || ! check_md5; then
    echo "Download or MD5 checksum check failed again."
    rm -f $measures_data_filename $measures_md5sum_filename
    cd .. && rmdir "tmp_$update_id"
    exit 1
  fi
fi

if ! tar jxf $measures_data_filename; then
  cd .. && rm -rf "tmp_$update_id"
  exit 1
fi

if ! cd ..; then exit 1; fi  # back to the $working_dir we had

# Remove earlier downloaded entries beyond the 3(+1 new tmp_) latest. ('ls' also sorts.)
old_update_ids=`ls -d $dir_prefix* 2> /dev/null | head -n -3`
if [ ! -z "$old_update_ids" ]; then
  rm -r $old_update_ids
  if [ $? -ne 0 ]; then
    echo "Failed to remove old measure table dir(s)."  # not fatal
  else
    echo "Removed old measure table dir(s):"
    echo "$old_update_ids"  # each on a new line
  fi
fi

# Make it available to the apply script to install/move it at an opportune moment.
if ! mv "tmp_$update_id" "$update_id"; then
  # leave the tmp_ dir in place for manual inspection
  echo "Error: failed to prepare measures table update 'tmp_$update_id'. Manual intervention likely required."
  exit 1
fi

echo "All Cool. Measures table update '$update_id' ready to apply."

