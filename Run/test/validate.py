#!/usr/bin/python
import sys
sys.path = sys.path + ["../src"]

from LOFAR.ParsetTester import ParsetTester
from LOFAR.LogValidators import NoErrors
from LOFAR.Locations import Locations
from LOFAR.Partitions import PartitionPsets
from LOFAR import Logger
from random import sample

parsetFile = "RTCP-validate.parset"

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
			default = "%s/VALIDATION-${TIMESTAMP}" % (os.getcwd(),),
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

  testgroup = OptionGroup(parser, "Tests to run (or all tests if nothing is specified)")
  testgroup.add_option( "--clock",
  			dest = "clock",
                        action = "store_true",
                        default = False,
			help = "run clock tests" )
  testgroup.add_option( "--oneoutput",
  			dest = "oneoutput",
                        action = "store_true",
                        default = False,
			help = "run tests for individual outputs" )
  testgroup.add_option( "--subbandrun",
  			dest = "subbandrun",
                        action = "store_true",
                        default = False,
			help = "run tests for varying number of subbands" )
  testgroup.add_option( "--beamrun",
  			dest = "beamrun",
                        action = "store_true",
                        default = False,
			help = "run tests for varying number of beams and subbands" )
  testgroup.add_option( "--stationrun",
  			dest = "stationrun",
                        action = "store_true",
                        default = False,
			help = "run tests for varying number of beams and stations" )
  testgroup.add_option( "--tabrun",
  			dest = "tabrun",
                        action = "store_true",
                        default = False,
			help = "run tests for merging stations" )
  parser.add_option_group( testgroup )

  # parse arguments
  (options, args) = parser.parse_args()

  if not options.partition:
    parser.print_help()
    sys.exit(1)

  Logger.initLogger()  

  for opt in dirgroup.option_list:
    Locations.setFilename( opt.dest, getattr( options, opt.dest ) )

  Locations.resolveAllPaths()

  run_all = not reduce( lambda x,y: x or getattr( options, y.dest ), testgroup.option_list, False )

  def initParset( name ):
    return ParsetTester( parsetFile, options.partition, name )

  def testParset( pt, validators ):
    pt.runParset()

    success = pt.validate( validators )

    if success and not options.keeplogs:
      pt.cleanup()

    return success  

  # clocks
  if run_all or options.clock:
    for clock in [160,200]:
      p = initParset( "%d MHz clock" % (clock,) )
      p.parset["Observation.sampleCock"] = clock

      if not testParset( p, [NoErrors()] ):
        sys.exit(1)

  # individual outputs
  if run_all or options.oneoutput:
    for output in ["CorrelatedData","CoherentStokes","IncoherentStokes"]:
      p = initParset( "output %s only" % (output,) )

      p.parset["Observation.output%s" % (output,)] = True

      if not testParset( p, [NoErrors()] ):
        sys.exit(1)

  # test 2 outputs, various number of subbands (for 2nd transpose)
  if run_all or options.subbandrun:
    nrBeams = 1

    for nrSubbands in [1,2,3,4,8,10,11,13,16,32,62,63,64,128,248]:
      if nrSubbands < nrBeams:
        continue

      p = initParset( "%d subbands" % (nrSubbands,) )
      p.setNrSubbands( nrSubbands )

      p.parset["Observation.outputCorrelatedData"] = True
      p.parset["Observation.outputCoherentStokes"] = True

      if not testParset( p, [NoErrors()] ):
        sys.exit(1)

  # test 2 outputs, various number of subbands (for 2nd transpose), multiple beams
  if run_all or options.beamrun:
    for nrBeams in [2,4,7,9,16,32,64]:
      for nrSubbands in [1,2,3,4,8,10,11,13,16,32,62,63,64,128,248]:
        if nrSubbands < nrBeams:
          continue

        p = initParset( "%d beams %d subbands" % (nrBeams,nrSubbands,) )
        p.setNrSubbands( nrSubbands )
        p.setNrPencilBeams( nrBeams )
        p.setNrStations( 2 )

        p.parset["Observation.outputCorrelatedData"] = True
        p.parset["Observation.outputCoherentStokes"] = True

        if not testParset( p, [NoErrors()] ):
          sys.exit(1)

  # test 2 outputs, various number of subbands (for 2nd transpose), multiple beams
  if run_all or options.stationrun:
    for nrStations in [2,4,8,16,32,62,64]:
      for nrBeams in [8,9,16]:
        if nrStations > len(PartitionPsets[options.partition]):
          continue

        p = initParset( "%d beams %d stations" % (nrBeams,nrStations,) )
        p.setNrStations( nrStations )
        p.setNrPencilBeams( nrBeams )

        p.parset["Observation.outputCorrelatedData"] = True
        p.parset["Observation.outputCoherentStokes"] = True

        if not testParset( p, [NoErrors()] ):
          sys.exit(1)

  if run_all or options.tabrun:
    # max nr stations
    nrStations = len(PartitionPsets[options.partition])
    for nrTabStations in xrange(2,nrStations+1):
      p = initParset( "%d merged stations" % (nrTabStations,) )
      p.setNrStations( nrStations )

      allStationNames = p.parset.stations

      # combine random stations
      tabList = sample( allStationNames,nrTabStations )
      p.parset["Observation.Beamformer[0].stationList"] = ",".join(tabList)

      if not testParset( p, [NoErrors()] ):
        sys.exit(1)
