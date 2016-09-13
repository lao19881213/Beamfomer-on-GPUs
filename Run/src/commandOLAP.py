#!/usr/bin/env python

from LOFAR import Logger
from logging import debug,info,warning,error,critical
from LOFAR.CommandClient import sendCommand
import sys
import socket

DRYRUN = False

if __name__ == "__main__":
  from optparse import OptionParser,OptionGroup
  import os
  import time

  # parse the command line
  parser = OptionParser( usage = """usage: %prog -P partition command
  
    where 'command' is one of the following:
   
    parset FILE         send the parset FILE to the correlator
    cancel OBSID        cancel or abort observation OBSID
    quit                stop the correlator after the last observation
  """ )

  opgroup = OptionGroup(parser, "Output" )
  opgroup.add_option( "-v", "--verbose",
  			dest = "verbose",
			action = "store_true",
			default = False,
  			help = "be verbose [%default]" )
  opgroup.add_option( "-q", "--quiet",
  			dest = "quiet",
			action = "store_true",
			default = False,
  			help = "be quiet [%default]" )
  parser.add_option_group( opgroup )

  hwgroup = OptionGroup(parser, "Hardware" )
  hwgroup.add_option( "-P", "--partition",
  			dest = "partition",
			type = "string",
  			help = "name of the BlueGene partition [%default]" )
  parser.add_option_group( hwgroup )

  # parse arguments
  (options, args) = parser.parse_args()

  # ========== Global options

  if not args or not options.partition:
    parser.print_help()
    sys.exit(1)

  if options.verbose:
    Logger.DEBUG = True

  if not options.quiet:
    DEBUG = True

  Logger.initLogger()  

  # connect and send command
  try:
    sendCommand( options.partition, " ".join(args) )
  except socket.error, msg:
    critical( "Socket error: %s" % (msg,) )
