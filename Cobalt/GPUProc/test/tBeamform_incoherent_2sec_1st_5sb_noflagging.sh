#!/bin/bash -x
TESTNAME=`basename "${0%%.sh}"`
./runctest.sh $TESTNAME

