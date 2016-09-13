# Bash functions used by the different GPU tests.
#
# This file must be source'd, not executed!

# Check if our system has a GPU installed.
haveGPU()
{
  if ! lspci | grep -E "VGA|3D" | grep -E "ATI|NVIDIA" > /dev/null
  then
    echo "No ATI/NVIDIA GPU card detected."
    return 1
  fi
  return 0
}

function error {
  echo "$@"
  exit 1
}

# set testdir to mirror srcdir
testdir=$PWD

function mklofarroot {
  DEST=$1

  # Make a fake LOFARROOT to be able to locate:
  #   $LOFARROOT/bin:   startBGL.sh, stopBGL.sh, rtcp, mpi_node_list
  #   $LOFARROOT/etc:   parset-additions.d/
  #   $LOFARROOT/share: gpu/kernels/*.cu
  # and the share/ directory for the GPU kernels
  mkdir -p $DEST || error "Cannot mkdir $DEST"
  export LOFARROOT=$PWD/$DEST
  echo "LOFARROOT=$LOFARROOT"

  ln -sfT $PWD/../src $LOFARROOT/bin || error "ln failed for bin"
  ln -sfT $srcdir/../etc $LOFARROOT/etc || error "ln failed for etc"
  ln -sfT $srcdir/../share $LOFARROOT/share || error "ln failed for share"
  mkdir -p $LOFARROOT/var/{run,log} || error "mkdir failed for var/run or var/log"
  mkdir -p $LOFARROOT/tmp || error "mkdir failed for tmp"

  # Let's work within our temp space
  cd $LOFARROOT/tmp || error "could not cd $LOFARROOT/tmp"

  # Crude hack for the apps to find their log_prop files. It will mess up
  # stdout without it.
  cp $srcdir/../src/Station/mpi_node_list.log_prop . || error "Could not copy mpi_node_list.log_prop"
  cp $srcdir/../src/rtcp.log_prop . || error "Could not copy rtcp.log_prop"

  # Be able to find all binaries through $PATH
  export PATH=$LOFARROOT/bin:$srcdir/../src:$srcdir/../src/scripts:$srcdir:$PATH
  echo "PATH=$PATH"
}

