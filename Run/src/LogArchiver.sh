#!/bin/bash
#
# LogArchiver.sh: a script to automatically archive old log files and parsets.
#
# This script is meant to be run fully automatic from a cron job
#

# cron jobs don't set $PATH, so construct our own.
export PATH=$PATH:/bin:/usr/bin:/opt/lofar/bin

source locations.sh

# source and destination for archiving
SRCDIR="$LOGDIR"
DESTDIR="$LOGBACKUPDIR"

# which file patterns to archive
PATTERNS=("CNProc.log.*" "IONProc.log.*" "startBGL.log.*" "*.parset")

# how old the last change to the file has to be (seconds)
MINAGE="7 * 24 * 60 * 60"




# expand patterns like "foo.*" to nothing if no file matches it
shopt -s nullglob

# create a sane environment
if [ "$SRCDIR" == "$DESTDIR" ]
then
  echo "Nothing to do: SRCDIR == DESTDIR == $SRCDIR"
  exit
fi

mkdir -p $DESTDIR || exit

# make a staging directory to avoid racing
# conditions if this script runs twice simultaneously
STAGEDIR=`mktemp -d "$SRCDIR/LogArchiver.sh-staging-XXXXXX"`

if [ -z "$STAGEDIR" ]
then
  echo "Could not create staging directory inside $SRCDIR"
  exit
fi

function age {
  # prints the age of the provided file, in seconds
  FILE=$1

  if [ -e "$FILE" ]
  then
    echo $((`date +%s` - `stat -c %Y "$FILE"`))
  else
    echo 0
  fi
}

function shouldarchive {
  # prints 1 iff the given file sould be archived
  FILE=$1

  if [ `age "$f"` -le $(($MINAGE)) ]
  then
    echo 0
    return
  fi

  echo 1
}

function archive {
  # considers the files matching the given pattern for archiveing
  FILES=$1

  echo ">>> request to archive $FILES"

  for f in $FILES
  do
    echo considering $f
    if [ `shouldarchive "$f"` -eq 1 ]
    then
      echo "++++" archiving: $f
      mv "$f" "$STAGEDIR" && gzip "$STAGEDIR/`basename "$f"`" && mv "$STAGEDIR/`basename "$f"`.gz" "$DESTDIR"
    else
      echo "----" not archiving: $f
    fi  
  done
}

# archive all files matching the pattern
for k in ${!PATTERNS[*]}
do
  archive "$SRCDIR/${PATTERNS[$k]}"
done

rmdir $STAGEDIR

