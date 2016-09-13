#!/usr/bin/python
import sys
sys.path = sys.path + ["../src"]

from LOFAR.ParsetTester import ParsetTester
from LOFAR.LogValidators import NoErrors,NoDrops,RealTime
from LOFAR.Locations import Locations
from LOFAR.Partitions import PartitionPsets
from LOFAR import Logger

if __name__ == "__main__":
  from optparse import OptionParser,OptionGroup
  import os
  import sys

  # parse the command line
  parser = OptionParser( usage = """usage: %prog -P partition [options]""" )

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
  opgroup.add_option( "-k", "--keeplogs",
                        dest = "keeplogs",
                        action = "store_true",
                        default = False,
                        help = "keep log files and data products of successful tests [%default]" )
  parser.add_option_group( opgroup )

  hwgroup = OptionGroup(parser, "Hardware" )
  hwgroup.add_option( "-P", "--partition",
                        dest = "partition",
                        type = "string",
                        help = "name of the BlueGene partition [%default]" )
  parser.add_option_group( hwgroup )

  dirgroup = OptionGroup(parser, "Directory and file locations")
  dirgroup.add_option( "--basedir",
  			dest = "basedir",
			default = Locations.files["basedir"],
			help = "base directory [%default]" )
  dirgroup.add_option( "--logdir",
  			dest = "logdir",
			default = "%s/test-${TIMESTAMP}" % (os.getcwd(),),
			help = "log directory (syntax: [host:]path) [%default]" )
  dirgroup.add_option( "--rundir",
  			dest = "rundir",
			default = "${LOGDIR}",
			help = "run directory [%default]" )
  dirgroup.add_option( "--cnproc",
  			dest = "cnproc",
			default = Locations.files["cnproc"],
			help = "CNProc executable [%default]" )
  dirgroup.add_option( "--ionproc",
  			dest = "ionproc",
			default = Locations.files["ionproc"],
			help = "IONProc executable [%default]" )
  parser.add_option_group( dirgroup )

  testgroup = OptionGroup(parser, "Test parameters" )
  testgroup.add_option( "--parset",
  			dest = "parset",
                        default = "RTCP.parset",
			help = "Parset to use [%default]" ),
  testgroup.add_option( "--name",
  			dest = "name",
                        default = "test",
			help = "Name of this test [%default]" ),
  testgroup.add_option( "-A", "--nrstations",
  			dest = "nrstations",
                        type = "int",
                        default = 0,
			help = "If >0, override the number of stations to use [%default]" )
  testgroup.add_option( "-B", "--nrbeams",
  			dest = "nrbeams",
                        type = "int",
                        default = 0,
			help = "If >0, override the number of pencil beams to use [%default]" )
  testgroup.add_option( "-S", "--nrsubbands",
  			dest = "nrsubbands",
                        type = "int",
                        default = 0,
			help = "If >0, override the number of subbands to use [%default]" )
  testgroup.add_option( "-r", "--runtime",
  			dest = "runtime",
                        default = 30,
                        type = "int",
			help = "Duration of the observation in seconds [%default]" )
  testgroup.add_option( "-o", "--option",
  			dest = "option",
                        action = "append",
                        default = [],
                        type = "string",
			help = "Additional parset key=value pairs." )

  parser.add_option_group( testgroup )

  valgroup = OptionGroup(parser, "Validation parameters" )
  valgroup.add_option( "-d", "--nodrops",
                        dest = "nodrops",
                        action = "store_true",
                        default = False,
                        help = "do not allow any data to be dropped [%default]" )

  # parse arguments
  (options, args) = parser.parse_args()

  if not options.partition:
    parser.print_help()
    sys.exit(1)

  Logger.initLogger()  

  for opt in dirgroup.option_list:
    Locations.setFilename( opt.dest, getattr( options, opt.dest ) )

  Locations.resolveAllPaths()

  pt = ParsetTester( options.parset, options.partition, options.name )
  if options.nrsubbands > 0: pt.setNrSubbands( options.nrsubbands )
  if options.nrbeams    > 0: pt.setNrPencilBeams( options.nrbeams )
  if options.nrstations > 0: pt.setNrStations( options.nrstations )
  for o in options.option:
    pt.parset.parse(o)

  pt.runParset( runtime=options.runtime, parsetstartdelay=50 )

  validators = [NoErrors(),RealTime()]
  if options.nodrops:
    validators.append( NoDrops() )

  success = pt.validate( validators )

  if success and not options.keeplogs:
    pt.cleanup()

  sys.exit(int(not success))

