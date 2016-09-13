#!/bin/bash -x

# Run a parset and compare the output to that in the reference_output directory.
# 
# Syntax: testParset.sh parset [-r reference-output-directory] [-g minimal-gpu-efficiency]

# Set defaults for options
REFDIR=
GPUEFFICIENCY=0

echo "Invoked as" "$0" "$@"

# Parse options
while getopts "r:g:" opt
do
  case $opt in
    r)
      REFDIR=$OPTARG
      ;;

    g)
      GPUEFFICIENCY=$OPTARG
      ;;

    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;

    :)
      echo "Option needs argument: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))

PARSET=$1

# Include some useful shell functions
. $srcdir/testFuncs.sh

# Some host info
echo "Running as `whoami`"
echo "Running on `hostname`"
echo "Working directory is `pwd`"

# Check for GPU
haveGPU || exit 3

# Check for input files
if [ ! -e /var/scratch/mol/test_sets ]
then
  echo "No input files found -- aborting test." >&2
  exit 3
fi

echo "Testing $PARSET"

OUTDIR=`basename "${PARSET%.parset}.output"`

function parse_logs
{
  NORMAL=$1
  PROFILED=$2

  test -r $NORMAL || return 1
  test -r $PROFILED || return 1

  # obtain wall time
  WALLTIME=`<$NORMAL perl -ne 'if (/Wall seconds spent.*?([0-9.]+)$/) { print $1; }'`
  # obtain GPU cost
  GPUCOST=`<$PROFILED perl -ne 'if (/GPU  seconds spent computing.*?([0-9.]+)$/) { print $1; }'`

  # log efficiency
  GPUUSAGE=`echo "scale=0;100*$GPUCOST/$WALLTIME" | bc -l`
  echo "Total processing time: $WALLTIME s"
  echo "GPU usage            : $GPUUSAGE %"

  if [ "$GPUUSAGE" -lt "$GPUEFFICIENCY" ]
  then
    echo "ERROR: GPU usage < $GPUEFFICIENCY% -- considering test a failure." >&2
    return 1
  fi

  return 0
}

(
  # Create fake LOFARROOT environment
  mklofarroot $OUTDIR

  # run correlator -- without profiling
  runObservation.sh -F -l 4 $PARSET > performance_normal.txt 2>&1 || error "Observation failed"

  # compare output
  if [ -n "$REFDIR" ]
  then
    # create script to accept output (ie. copy it to the source dir for check in)
    echo "#!/bin/bash
    cp `pwd`/*.MS $REFDIR" > accept_output
    chmod a+x accept_output

    # GCC on x86_64 has std::numeric_limits<float>::epsilon() = 1.192092896e-07f
    numfp32eps=\(1.192092896/10000000\)

    # Generally (tCorrelate_*), the first 5 decimals are ok; occasionally, the 5th is off.
    # For the tCorrelate tests, 16*num_lim<float>::eps() is not enough.
    # Taking 32*..., we still get a few dozen miscomparisons, so resort to 64.0
    #
    # Try bigger epsilons as well to see how big the error actually is.
    for eps_factor in 1024.0 512.0 256.0 128.0 64.0 32.0 16.0 8.0
    do
      EPSILON=$(echo $eps_factor \* $numfp32eps | bc -l)

      for f in *.MS
      do
        $testdir/cmpfloat $EPSILON `pwd`/$f $REFDIR/$f || error "Output does not match reference for eps_factor=$eps_factor"
      done
    done
  fi

  # run correlator -- with profiling
  runObservation.sh -F -l 4 -p $PARSET > performance_profiled.txt 2>&1 || error "Profiling observation failed"

  # check logs
  parse_logs performance_normal.txt performance_profiled.txt || error "Could not parse log files"

  # toss output if everything is ok, but do not fail test if removal fails
  rm -rf $testdir/$OUTDIR || true # Comment this line for output
) || exit 1

