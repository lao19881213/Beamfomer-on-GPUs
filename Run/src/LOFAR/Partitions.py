#!/usr/bin/env python

__all__ = [ "PartitionPsets" ]

import os
import sys

# allow ../util to be found, a bit of a hack
sys.path += [os.path.abspath(os.path.dirname(__file__)+"/..")]

# PartitionPsets:	A dict which maps partitions to I/O node IP addresses.
# the pset hierarchy is is analogue to:
# R00-M0-N00-J00 = R00-M0-N00-J00-16 consists of a single pset
# R00-M0-N00-32  = R00-M0-N00-J00 + R00-M0-N00-J01
# R00-M0-N00-64  = R00-M0-N00-32  + R00-M0-N01-32
# R00-M0-N00-128 = R00-M0-N00-64  + R00-M0-N02-64
# R00-M0-N00-256 = R00-M0-N00-128 + R00-M0-N04-128
# R00-M0         = R00-M0-N00-256 + R00-M0-N08-256
# R00            = R00-M0 + R00-M1

# LOFARTEST      = R01-M0-N00-J00 + R01-M0-N08-J00

PartitionPsets = {}
for R in xrange(3):
  rack = "R%02d" % R
  for M in xrange(2):
    midplane = "%s-M%01d" % (rack,M)

    # individual psets
    for N in xrange(16):
       nodecard = "%s-N%02d" % (midplane,N)
       for J in xrange(2):
         # ip address for this pset
         ip = "10.170.%d.%d" % (R,(1+M*128+N*4+J))

         pset = "%s-J%02d" % (nodecard,J)
         if R == 0: PartitionPsets[pset] = [ip] # single psets without -16 suffix only work on R00
         PartitionPsets["%s-16" % (pset,)] = [ip]

    # groups smaller than a midplane
    for groupsize in (1,2,4,8):
      for N in xrange(0,16,groupsize):
        nodecard = "%s-N%02d" % (midplane,N)

        PartitionPsets["%s-%d" % (nodecard,32*groupsize)] = sum( [
          PartitionPsets["%s-N%02d-J00-16" % (midplane,x)] + PartitionPsets["%s-N%02d-J01-16" % (midplane,x)]
         for x in xrange( N, N+groupsize) ], [] )

    # a midplane
    PartitionPsets[midplane] = PartitionPsets["%s-N00-256" % midplane] + PartitionPsets["%s-N08-256" % midplane]

  # a rack
  PartitionPsets[rack] = PartitionPsets["%s-M0" % rack] + PartitionPsets["%s-M1" % rack]

PartitionPsets["R00R01"] = PartitionPsets["R00"] + PartitionPsets["R01"]  
PartitionPsets["LOFARTEST"] = PartitionPsets["R01-M0-N00-J00-16"] + PartitionPsets["R01-M0-N08-J00-16"]  

if __name__ == "__main__":
  from optparse import OptionParser,OptionGroup
  import sys

  # parse the command line
  parser = OptionParser( "usage: %prog [options] partition" )
  parser.add_option( "-l", "--list",
  			dest = "list",
			action = "store_true",
			default = False,
  			help = "list the psets in the partition" )

  # parse arguments
  (options, args) = parser.parse_args()
  errorOccurred = False

  if not args:
    parser.print_help()
    sys.exit(0)

  for partition in args:
    assert partition in PartitionPsets,"Partition unknown: %s" % (partition,)

    if options.list:
      # print the psets of a single partition
      for ip in PartitionPsets[partition]:
        print ip

    sys.exit(int(errorOccurred))

