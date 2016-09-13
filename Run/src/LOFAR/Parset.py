#!/usr/bin/python
import sys,os

# allow ../util to be found, a bit of a hack
sys.path += [(os.path.dirname(__file__) or ".")+"/.."]

import util.Parset
import os
from itertools import count
from Partitions import PartitionPsets
from Stations import Stations, overrideRack
from RingCoordinates import RingCoordinates
from util.dateutil import parse,format,parseDuration,timestamp
from logging import error,warn
import math
from sets import Set

# if True, reroute all traffic from first IO node to all but the last storage node,
# and the rest to the last storage node
PERFORMANCE_TEST = False

NRRSPBOARDS=4
NRBOARDBEAMLETS=61

class Parset(util.Parset.Parset):
    def __init__(self):
        util.Parset.Parset.__init__(self)

	self.stations = []
	self.storagenodes = []
	self.partition = ""
	self.psets = []

        self.filename = ""

    def applyAntennaSet( self, station, antennaset = None ):
      if antennaset is None:
        antennaset = self["Observation.antennaSet"]

      if antennaset == "":
        # useful for manually entered complete station names like CS302HBA1
        suffix = [""]
      elif antennaset in ["LBA_INNER","LBA_OUTER","LBA_X","LBA_Y","LBA_SPARSE_EVEN","LBA_SPARSE_ODD"]:
        suffix = ["LBA"]
      elif station.startswith("CS"):
        if antennaset in ["HBA_ZERO","HBA_ZERO_INNER"]:
          suffix = ["HBA0"]
        elif antennaset in ["HBA_ONE","HBA_ONE_INNER"]:
          suffix = ["HBA1"]
        elif antennaset in ["HBA_JOINED","HBA_JOINED_INNER"]:
          suffix = ["HBA"]
        elif antennaset in ["HBA_DUAL","HBA_DUAL_INNER"]:
          suffix = ["HBA0","HBA1"]
        else: 
          assert false, "Unknown antennaSet: %s" % (antennaset,)
      else:  
        suffix = ["HBA"]

      return "+".join(["%s%s" % (station,s) for s in suffix])


    def setFilename( self, filename ):
        self.filename = filename

    def getFilename( self ):
        return self.filename

    def save(self):
        self.writeFile( self.filename )

    def distillStations(self, key="Observation.VirtualInstrument.stationList"):
        """ Distill station names to use from the parset file and return them. """

        if key not in self:
          return "+".join(self.get('OLAP.storageStationNames',""))

        # translate station name + antenna set to CEP comprehensable names
        antennaset = self["Observation.antennaSet"]

        return "+".join( [self.applyAntennaSet(s) for s in self[key]] )

    def distillPartition(self, key="OLAP.CNProc.partition"):
        """ Distill partition to use from the parset file and return it. """

        if key not in self:
          return ""

        return self[key]

    def distillStorageNodes(self, key="OLAP.Storage.hosts"):
        """ Distill storage nodes to use from the parset file and return it. """

        if key in self:
          return self.getStringVector(key)
  
        outputnames = ["Correlated","Beamformed","Trigger"] + ["CoherentStokes","IncoherentStokes"] # still parse Coherent and Incoherent because the scheduler still sets them. While we collapse them into Beamformed later on, this code is needed before that
        locationkeys = ["Observation.DataProducts.Output_%s.locations" % p for p in outputnames]

        storagenodes = set()

        for k in locationkeys:
          if k not in self:
            continue

          locations = self.getStringVector(k)
          hostnames = [x.split(":")[0] for x in locations]
          storagenodes.update( set(hostnames) )

        return list(storagenodes)

    def postRead(self):
        """ Distill values for our internal variables from the parset. """

        # partition
        partition = self.distillPartition()
        if partition:
          self.setPartition( partition )

        if self.partition and self.partition != "R00R01" and self.partition != "LOFARTEST":  
          overrideRack( Stations, int(self.partition[2]) )

        # storage nodes
        storagenodes = self.distillStorageNodes() or []
        self.setStorageNodes( storagenodes )

        # stations
        stationStr = self.distillStations()
        if stationStr:
          stationList = Stations.parse( stationStr )
          self.setStations( stationList )

    def addMissingKeys(self):
        """ Sets some default values which SAS does not yet contain. """

        # meta data
        self.setdefault("Observation.ObserverName","unknown")
        self.setdefault("Observation.ProjectName","unknown")

        self.setdefault("Observation.DataProducts.Output_Trigger.enabled",False)

        if 'OLAP.PPF' in self:
          if not self['OLAP.PPF']:
            self["Observation.channelsPerSubband"] = 1
        else:  
          self['OLAP.PPF'] = int(self["Observation.channelsPerSubband"]) > 1

        for k in ["OLAP.CNProc_CoherentStokes.channelsPerSubband", "OLAP.CNProc_IncoherentStokes.channelsPerSubband"]:
          if k not in self or int(self[k]) == 0:
            self[k] = self["Observation.channelsPerSubband"]
        self.setdefault('Observation.DataProducts.Output_Beamformed.namemask','L${OBSID}_SAP${SAP}_B${BEAM}_S${STOKES}_P${PART}_bf.raw')
        self.setdefault('Observation.DataProducts.Output_Correlated.namemask','L${OBSID}_SB${SUBBAND}_uv.MS')
        self.setdefault('Observation.DataProducts.Output_Trigger.namemask','L${OBSID}_SAP${SAP}_B${BEAM}_S${STOKES}_P${PART}_bf.trigger')
	self.setdefault('OLAP.dispersionMeasure', 0);

        self.setdefault('Observation.DataProducts.Output_Beamformed.dirmask','L${YEAR}_${OBSID}')
        self.setdefault('Observation.DataProducts.Output_Correlated.dirmask','L${YEAR}_${OBSID}')
        self.setdefault('Observation.DataProducts.Output_Trigger.dirmask','L${YEAR}_${OBSID}')

        # default beamlet settings, derived from subbandlist, for development
	if "Observation.subbandList" in self:
	  nrSubbands = len(self.getInt32Vector("Observation.subbandList"))
        else:
          nrSubbands = NRRSPBOARDS*NRBOARDBEAMLETS

        if "Observation.nrBitsPerSample" in self:
          bitmode = int(self["Observation.nrBitsPerSample"])
        elif "OLAP.nrBitsPerSample" in self:
          bitmode = int(self["OLAP.nrBitsPerSample"])
        else:
          bitmode = 16

        if "Observation.nrSlotsInFrame" in self:
          slots = int(self["Observation.nrSlotsInFrame"])
        else:
          slots = 16/bitmode * NRBOARDBEAMLETS

        self.setdefault("Observation.subbandList",  [151+s for s in xrange(nrSubbands)])  
	self.setdefault("Observation.beamList",     [0     for s in xrange(nrSubbands)])

    def convertSASkeys(self):
        """ Convert keys generated by SAS to those used by OLAP. """

        def delIfEmpty( k ):
          if k in self and not self[k]:
            del self[k]

        # SAS cannot omit keys, so assume that empty keys means 'use default'
        delIfEmpty( "OLAP.CNProc.phaseOnePsets" )
        delIfEmpty( "OLAP.CNProc.phaseTwoPsets" )
        delIfEmpty( "OLAP.CNProc.phaseThreePsets" )

        # make sure these values will be recalculated in finalise()
        del self['OLAP.IONProc.integrationSteps']
        del self['OLAP.CNProc.integrationSteps']

        # convert pencil rings and fly's eye to more coordinates
        for b in count():
          if "Observation.Beam[%s].angle1" % (b,) not in self:
            break

          self.setdefault("Observation.Beam[%s].nrTabRings" % (b,),0)
          self.setdefault("Observation.Beam[%s].tabRingSize" % (b,),0.0)

          dirtype = self["Observation.Beam[%s].directionType" % (b,)]  
          center_angle1 = float(self["Observation.Beam[%s].angle1" % (b,)])
          center_angle2 = float(self["Observation.Beam[%s].angle2" % (b,)])
          dm = int(self.get("OLAP.dispersionMeasure",0))

          nrrings = int(self["Observation.Beam[%s].nrTabRings" % (b,)]) 
          width   = float(self["Observation.Beam[%s].tabRingSize" % (b,)]) 
          ringcoordinates = RingCoordinates( nrrings, width, (center_angle1, center_angle2), dirtype )
          ringset = [
            { "angle1": angle1,
              "angle2": angle2,
              "directionType": dirtype,
              "dispersionMeasure": dm,
              "stationList": [],
              "specificationType": "ring",
              "coherent": True,
            } for (angle1,angle2) in ringcoordinates.coordinates()
          ]

          flyseyeset = []

          if self.getBool("OLAP.PencilInfo.flysEye"):
	    allStationNames = [st.getName() for st in self.stations]

            for s in allStationNames:
              flyseyeset.append(
                { "angle1": 0,
                  "angle2": 0,
                  "directionType": dirtype,
                  "dispersionMeasure": dm,
                  "stationList": [s],
                  "specificationType": "flyseye",
                  "coherent": True,
                }
              )

          manualset = []

          for m in count():
            if "Observation.Beam[%s].TiedArrayBeam[%s].angle1" % (b,m) not in self:
              break

            manualset.append(
              { "angle1": self["Observation.Beam[%s].TiedArrayBeam[%s].angle1" % (b,m)],
                "angle2": self["Observation.Beam[%s].TiedArrayBeam[%s].angle2" % (b,m)],
                "directionType": self["Observation.Beam[%s].TiedArrayBeam[%s].directionType" % (b,m)],
                "dispersionMeasure": self["Observation.Beam[%s].TiedArrayBeam[%s].dispersionMeasure" % (b,m)],
                "stationList": [],
                "specificationType": "manual",
                "coherent": self["Observation.Beam[%s].TiedArrayBeam[%s].coherent" % (b,m)],
              }
            )

          # first define the rings, then the manual beams (which thus get shifted in number!)
          allsets = manualset + ringset + flyseyeset
          for m,s in enumerate(allsets):
            prefix = "Observation.Beam[%s].TiedArrayBeam[%s]" % (b,m)

            for k,v in s.iteritems():
              self["%s.%s" % (prefix,k)] = v

          self["Observation.Beam[%s].nrTiedArrayBeams" % (b,)] = len(allsets)    

        # SAS specifies beams differently
        if "Observation.subbandList" not in self:
          # convert beam configuration
          allSubbands = []

	  for b in count():
            if "Observation.Beam[%s].angle1" % (b,) not in self:
              break

	    beamSubbands = self.getInt32Vector("Observation.Beam[%s].subbandList" % (b,)) # the actual subband number (0..511)

            for subband in beamSubbands:
              allSubbands.append( {
                "beam":     b,
                "subband":  subband,
              } )


          # order subbands according to beamlet id, for more human-friendly reading
          sortedSubbands = sorted( allSubbands )

          # populate OLAP lists
	  self["Observation.subbandList"]  = [s["subband"] for s in sortedSubbands]
	  self["Observation.beamList"]     = [s["beam"] for s in sortedSubbands]

        # The Scheduler creates three lists of files (for beamformed, coherent and incoherent),
        # but we collapse this into one list (beamformed)

        def getlist( dataproduct ):
          enabled = self.getBool("Observation.DataProducts.Output_%s.enabled" % (dataproduct,), False)

          if True or enabled: # scheduler can still set filenames without the corresponding enabled flag
            filenames = self.get("Observation.DataProducts.Output_%s.filenames" % (dataproduct,), [])
            locations = self.get("Observation.DataProducts.Output_%s.locations" % (dataproduct,), [])
          else:
            filenames = []
            locations = []

          return (filenames, locations)

        beamformedFiles  = getlist("Beamformed")
        coherentFiles    = getlist("CoherentStokes")
        incoherentFiles  = getlist("IncoherentStokes")

        # either coherent or beamformed are set as they are mutually exclusive
        if coherentFiles == ([], []):
          coherentFiles = beamformedFiles

        # if nothing is set, the filenames and locations are generated preWrite
        if coherentFiles != ([], []) or incoherentFiles != ([], []):
          # this will be the final list
          beamformedFiles = ([], [])  

          # reconstruct the full list  
          for b in count():
            if "Observation.Beam[%s].angle1" % (b,) not in self:
              break

            for t in count():
              if "Observation.Beam[%s].TiedArrayBeam[%s].angle1" % (b,t) not in self:
                break

              coherent = self.getBool("Observation.Beam[%s].TiedArrayBeam[%s].coherent" % (b,t))

              nrstokes = self.getNrStokes(coherent)
              nrparts  = self.getNrParts(b, coherent)

              for s in xrange(nrstokes):
                for p in xrange(nrparts):
                  if coherent:
                    filename = coherentFiles[0].pop(0)
                    location = coherentFiles[1].pop(0)
                  else:
                    filename = incoherentFiles[0].pop(0)
                    location = incoherentFiles[1].pop(0)

                  beamformedFiles[0].append(filename)
                  beamformedFiles[1].append(location)

          self["Observation.DataProducts.Output_Beamformed.enabled"] = True
          self["Observation.DataProducts.Output_Beamformed.filenames"] = beamformedFiles[0]
          self["Observation.DataProducts.Output_Beamformed.locations"] = beamformedFiles[1]

        if beamformedFiles != ([], []):
          # fix a scheduler bug causing this boolean not to be set
          self["Observation.DataProducts.Output_Beamformed.enabled"] = True


    def preWrite(self):
        """ Derive some final keys and finalise any parameters necessary
	    before writing the parset to disk. """

        self.convertSASkeys();
        self.addMissingKeys();

        # Versioning info
        self["OLAP.BeamsAreTransposed"] = True
        self["OLAP.IncoherentStokesAreTransposed"] = True

	# TODO: we use self.setdefault, but this can create inconsistencies if we
	# set one value but not the other in a pair of interdependent parameters.
	# This should possibly be detected in the check routine, but it seems
	# sloppy to let it pass through here unnoticed.

	# tied-array beam forming
        superStations = []

        for index in count():
          if "Observation.Beamformer[%s].stationList" % (index,) not in self:
            break

	  stations = self.getStringVector('Observation.Beamformer[%s].stationList' % (index,))

          stations = [self.applyAntennaSet(st) for st in stations]

          superStations.append(stations)
	
	if superStations != []:
	  # tied-array beamforming will occur

	  # add remaining stations to the list
	  allStationNames = [st.getName() for st in self.stations]
          beamFormedStations = sum(superStations, [])
          individualStations = [st for st in allStationNames if st not in beamFormedStations]


          allTabs = superStations + [[st] for st in individualStations]
          # sorting is important: because the original station list is sorted, a sorted tabList makes sure that no slot is overwritten before it is needed (data is always generated before or at the slot of the source station)
          allTabs.sort()

	  def findTabStation( st ):
	    for nr,tab in enumerate(allTabs):
              if st in tab:
                return nr

	  tabList = map( findTabStation, allStationNames )

          # make sure this tabList can be processed by going from element 0 to n-1 (dest slot is always at or after source slot)
          for st,nr in enumerate(tabList):
            assert st >= nr, "Station %s is at position %u in the station list but at position %u in the tab list, which could lead to data corruption" % (allStationNames[st],st,nr)
	   
	  self.setdefault('OLAP.tiedArrayStationNames', ["+".join(x) for x in allTabs])
	  self.setdefault('OLAP.CNProc.tabList', tabList)
        else:
          # no super-station beam forming
	  self.setdefault('OLAP.tiedArrayStationNames', [])
	  self.setdefault('OLAP.CNProc.tabList', [])

	# input flow configuration
	for station in self.stations:
	  self.setdefault('PIC.Core.Station.%s.RSP.ports' % (station.name,), station.inputs)
	  
          stationName = station.name.split("_")[0] # remove specific antenna or array name (_hba0 etc) if present
          self.setdefault("PIC.Core.%s.position" % (stationName,), self["PIC.Core.%s.phaseCenter" % (stationName,)])

	for pset in xrange(len(self.psets)):
	  self.setdefault('PIC.Core.IONProc.%s[%s].inputs' % (self.partition,pset), [
	    "%s/RSP%s" % (station.name,rsp) for station in self.stations
	                                      if station.getPsetIndex(self.partition) == pset
	                                    for rsp in xrange(len(station.inputs))] )


	# output flow configuration
        self['OLAP.storageStationNames'] = [s.name for s in self.stations]
        self['Observation.VirtualInstrument.stationList'] = [s.name for s in self.stations]

        self.setdefault('OLAP.OLAP_Conn.IONProc_Storage_Transport','TCP');
        self.setdefault('OLAP.OLAP_Conn.IONProc_CNProc_Transport','FCNP');

	# subband configuration
	if "Observation.subbandList" in self:
	  nrSubbands = len(self.getInt32Vector("Observation.subbandList"))
        else:
          nrSubbands = 248

        for nrBeams in count():
          if "Observation.Beam[%s].angle1" % (nrBeams,) not in self:
            break

          self.setdefault("Observation.Beam[%s].target" % (nrBeams,),"")
          self.setdefault("Observation.Beam[%s].directionType" % (nrBeams,),"J2000")

        self.setdefault('Observation.nrBeams', nrBeams)

	# Pset configuration
	self['OLAP.CNProc.partition'] = self.partition
        self['OLAP.IONProc.psetList'] = self.psets

	nrPsets = len(self.psets)
	nrStorageNodes = self.getNrUsedStorageNodes()
        nrBeamFiles = self.getNrBeamFiles()

        # set storage hostnames
        self["OLAP.Storage.hosts"] = self.storagenodes[:]

	self.setdefault('OLAP.nrPsets', nrPsets)
	self.setdefault('OLAP.CNProc.phaseOnePsets', [s.getPsetIndex(self.partition) for s in self.stations])
	self.setdefault('OLAP.CNProc.phaseTwoPsets', range(nrPsets))
        if self.phaseThreeExists():
	  self.setdefault('OLAP.CNProc.phaseThreePsets', self['OLAP.CNProc.phaseTwoPsets'])
        else:  
	  self.setdefault('OLAP.CNProc.phaseThreePsets', [])

        if 'OLAP.CNProc.usedCoresInPset' in self:
          cores = self.getInt32Vector("OLAP.CNProc.usedCoresInPset")
        else:
          cores = range(64)

        self.setdefault('OLAP.CNProc.phaseOneTwoCores',cores)  
        self.setdefault('OLAP.CNProc.phaseThreeCores',cores)

        # what will be stored where?
        # outputSubbandPsets may well be set before finalize()
	subbandsPerPset = int( math.ceil(1.0 * nrSubbands / max( 1, len(self["OLAP.CNProc.phaseTwoPsets"]) ) ) )
        beamsPerPset = int( math.ceil(1.0 * nrBeamFiles / max( 1, len(self["OLAP.CNProc.phaseThreePsets"]) ) ) )

        def _sn( sb, sb_pset0, nr_sb ):
          if nrStorageNodes <= 1:
            return 0

          if PERFORMANCE_TEST:
            if sb < sb_pset0:
              return sb % (nrStorageNodes - 1)
            else:
              return nrStorageNodes - 1
          else:
            return sb//int(math.ceil(1.0 * nr_sb/nrStorageNodes));

        self.setdefault('OLAP.storageNodeList',[_sn(i,subbandsPerPset,nrSubbands) for i in xrange(nrSubbands)])

	self.setdefault('OLAP.PencilInfo.storageNodeList',[_sn(i,beamsPerPset,nrBeamFiles) for i in xrange(nrBeamFiles)])

        self.setdefault('OLAP.Storage.targetDirectory','/data')

        # generate filenames to produce - phase 2
        nodelist = self.getInt32Vector( "OLAP.storageNodeList" );
        products = ["Correlated"]
        outputkeys = ["Correlated"]

        for p,o in zip(products,outputkeys):
          outputkey    = "Observation.DataProducts.Output_%s.enabled" % (o,)
          if not self.getBool(outputkey):
            continue

          maskkey      = "Observation.DataProducts.Output_%s.namemask" % p
          mask         = self["OLAP.Storage.targetDirectory"] + "/" + self["Observation.DataProducts.Output_%s.dirmask" % p] + "/" + self[maskkey]
          locationkey  = "Observation.DataProducts.Output_%s.locations" % p
          filenameskey = "Observation.DataProducts.Output_%s.filenames" % p

          if locationkey in self and filenameskey in self:
            continue

          paths = [ self.parseMask( mask, subband = i ) for i in xrange(nrSubbands) ]
          filenames = map( os.path.basename, paths )
          dirnames = map( os.path.dirname, paths )

          if self.storagenodes:
            locations = [ "%s:%s" % (self.storagenodes[nodelist[i]], dirnames[i]) for i in xrange(nrSubbands) ]
          else:
            locations = [ "" for i in xrange(nrSubbands) ]

          self.setdefault( locationkey, locations )
          self.setdefault( filenameskey, filenames )

        # generate filenames to produce - phase 3
        nodelist = self.getInt32Vector( "OLAP.PencilInfo.storageNodeList" );
        products = ["Beamformed","Trigger"]
        outputkeys = ["Beamformed","Trigger"]

        for p,o in zip(products,outputkeys):
          outputkey    = "Observation.DataProducts.Output_%s.enabled" % (o,)
          if not self.getBool(outputkey):
            continue

          maskkey      = "Observation.DataProducts.Output_%s.namemask" % p
          mask         = self["OLAP.Storage.targetDirectory"] + "/" + self["Observation.DataProducts.Output_%s.dirmask" % p] + "/" + self[maskkey]
          locationkey  = "Observation.DataProducts.Output_%s.locations" % p
          filenameskey = "Observation.DataProducts.Output_%s.filenames" % p

          if locationkey in self and filenameskey in self:
            continue

          # python iterates over last 'for' first!
          # this is the order generated by the IO nodes (see IONProc/src/Job.cc)
          paths = [ self.parseMask( mask, sap = sap, beam = b, stokes = s, part = p )
                    for sap in xrange(self.getNrSAPs())
                    for b in xrange(self.getNrBeams(sap))
                    for s in xrange(self.getNrStokes(self.isCoherent(sap, b)))
                    for p in xrange(self.getNrParts(sap, self.isCoherent(sap, b)))
                  ]
          filenames = map( os.path.basename, paths )
          dirnames = map( os.path.dirname, paths )

          if self.storagenodes:
            locations = [ "%s:%s" % (self.storagenodes[nodelist[i]], dirnames[i]) for i in xrange(self.getNrBeamFiles())]
          else:
            locations = [ "" for i in xrange(nrSubbands) ]

          self.setdefault( locationkey, locations )
          self.setdefault( filenameskey, filenames )

	# calculation configuration

        # integration times of CNProc and IONProc, based on self.integrationtime
        # maximum amount of time CNProc can integrate due to memory constraints
        if self.phaseThreeExists():
          maxCnIntegrationTime = 0.25
          defaultCnIntegrationTime = 0.25
        else:
          maxCnIntegrationTime = 1.0
          defaultCnIntegrationTime = 1.0

        # (minimal) number of times the IONProc will have to integrate
        integrationtime = float( self["OLAP.Correlator.integrationTime"] ) or defaultCnIntegrationTime
        ionIntegrationSteps = int(math.ceil(integrationtime / maxCnIntegrationTime))
        self.setdefault('OLAP.IONProc.integrationSteps', ionIntegrationSteps)

        # the amount of time CNProc will integrate, translated into samples
        cnIntegrationTime = integrationtime / int(self["OLAP.IONProc.integrationSteps"])
        nrSamplesPerSecond = int(self['Observation.sampleClock']) * 1e6 / 1024 / int(self['Observation.channelsPerSubband'])

        def gcd( a, b ):
          while b > 0:
            a, b = b, a % b
          return a

        def lcm( a, b ):
          if b == 0: return a
          if a == 0: return b
          return a * b / gcd(a, b)

        def lcmlist( l ):
          return reduce(lcm, l, 1)

        def roundTo( x, y ):
          """ Round x to a multiple of y. """
          return max(int(round(x/y))*y,y)

        def increase_factors(n):
          # increase the factors of n; returns a value close to n (<10% off for n<=195312)
          if n < 4: return n

          factors = []

          while n > 1:
            for f in [2,3,5,7]:
              if n % f == 0:
                factors += [f]
                n /= f
                break
            else:
              n += 1
              factors += [3]
              n /= 3

          prod = lambda l: reduce(lambda x,y: x*y,l,1)
          return prod(factors)

        # cnIntegrationSteps MUST be a multiple of these values
        forced_factors = lcmlist( [
          16,
          int(self["OLAP.CNProc_CoherentStokes.timeIntegrationFactor"]),
          int(self["OLAP.CNProc_IncoherentStokes.timeIntegrationFactor"]),
          int(self.get("OLAP.CNProc.dedispersionFFTsize",1)),
          ] )

        if self.getBool("Observation.DataProducts.Output_Correlated.enabled"):
          # if doing imaging, don't mess with the block size too much
          cnIntegrationSteps = roundTo( nrSamplesPerSecond * cnIntegrationTime, forced_factors )
        else:  
          # make sure that the remainder is easily factorisable for easier post-processing
          cnIntegrationSteps = forced_factors * increase_factors( int(round(nrSamplesPerSecond * cnIntegrationTime / forced_factors)) )

        cnIntegrationSteps = max(forced_factors, cnIntegrationSteps)

        self.setdefault('OLAP.CNProc.integrationSteps', cnIntegrationSteps)

    def setStations(self,stations):
	""" Set the array of stations to use (used internally). """

        def name( s ):
          try:
            return s.name
          except:
            return s
	
	self.stations = sorted( stations, cmp=lambda x,y: cmp(name(x),name(y)) )

    def setPartition(self,partition):
	""" Define the partition to use. """

	assert partition in PartitionPsets, "Partition %s unknown. Run LOFAR/Partitions.py to get a list of valid partitions." % (partition,)
	
	self.partition = partition
	self.psets = PartitionPsets[partition]

    def setStorageNodes(self,storagenodes):
	""" Define the list of storage nodes to use. """

        # do not resolve host names, since the resolve depends on the need (i.e. NIC needed)
	self.storagenodes = sorted(storagenodes)

        # OLAP needs IP addresses from the backend
        self["OLAP.Storage.hosts"] = self.storagenodes[:]

    def setObsID(self,obsid):
        self.setdefault("Observation.ObsID", obsid)

    def getObsID(self):
        if "Observation.ObsID" not in self:
          return None

        return int(self["Observation.ObsID"])

    def getNrUsedStorageNodes(self):
        return len(self.storagenodes)

    def parseMask( self, mask, sap = 0, subband = 0, beam = 0, stokes = 0, part = 0 ):
      """ Fills a mask. """

      assert "Observation.ObsID" in self, "Observation ID not generated yet."

      # obtain settings
      date = parse( self["Observation.startTime"] ).timetuple()

      # fill in the mask
      datenames = [ "YEAR", "MONTH", "DAY", "HOURS", "MINUTES", "SECONDS" ] # same order as in time tuple
      for index,d in enumerate(datenames):
        mask = mask.replace( "${%s}" % d, "%02d" % (date[index],) )

      mask = mask.replace( "${OBSID}", "%05d" % (self.getObsID(),) )
      mask = mask.replace( "${MSNUMBER}", "%05d" % (self.getObsID(),) )
      mask = mask.replace( "${SUBBAND}", "%03d" % (subband,) )
      mask = mask.replace( "${SAP}", "%03d" % (sap,) )
      mask = mask.replace( "${PART}", "%03d" % (part,) )
      mask = mask.replace( "${BEAM}", "%03d" % (beam,) )
      mask = mask.replace( "${STOKES}", "%01d" % (stokes,) )

      return mask

    def setStartStopTime( self, starttime, stoptime ):
      start = timestamp( parse( starttime ) )
      stop  = timestamp( parse( stoptime ) )

      self["Observation.startTime"] = format( start )
      self["Observation.stopTime"] = format( stop )

    def setStartRunTime( self, starttime, duration ):
      start = timestamp( parse( starttime ) )
      stop  = start + parseDuration( duration ) 

      self["Observation.startTime"] = format( start )
      self["Observation.stopTime"] = format( stop )

    def getNrSAPs( self ):
      return int(self["Observation.nrBeams"])

    def getNrSubbands( self, sap ):
      return sum([1 for s in self.getInt32Vector("Observation.beamList") if s == sap])

    def isCoherent( self, sap, tab ):
      return self.getBool("Observation.Beam[%s].TiedArrayBeam[%s].coherent" % (sap, tab))

    def getNrParts( self, sap, coherent ):
      if coherent:
        prefix = "OLAP.CNProc_CoherentStokes"
      else:  
        prefix = "OLAP.CNProc_IncoherentStokes"

      subbands = self.getNrSubbands(sap)
      subbandsPerFile = int(self.get("%s.subbandsPerFile" % (prefix,),subbands))
      return int(math.ceil(1.0 * subbands / subbandsPerFile))

    def getNrBeams( self, sap ):
      return self["Observation.Beam[%u].nrTiedArrayBeams" % (sap,)]

    def getNrMergedStations( self ):
      tabList = self["OLAP.CNProc.tabList"]

      if not tabList:
        return len(self.stations)

      return max(tabList) + 1  

    def getNrStokes( self, coherent ):
      if coherent:
        prefix = "OLAP.CNProc_CoherentStokes"
      else:  
        prefix = "OLAP.CNProc_IncoherentStokes"

      return len(self["%s.which" % (prefix,)]) # todo: recombine Xi+Xr and Yi+Yr for trigger

    def getNrBeamFiles( self ):
      files = 0
      for sap in xrange(self.getNrSAPs()):
        for tab in xrange(self.getNrBeams(sap)):
          coherent = self.isCoherent(sap, tab)

          files += self.getNrStokes(coherent) * self.getNrParts(sap, coherent)

      return files

    def phaseThreeExists( self ):  
      output_keys = [
        "Observation.DataProducts.Output_Beamformed.enabled",
        "Observation.DataProducts.Output_Trigger.enabled",
      ]

      for k in output_keys:
        if k in self and self.getBool(k):
          return True

      return False

    def phaseThreePsetDisjunct( self ):
      phase1 = set(self.getInt32Vector("OLAP.CNProc.phaseOnePsets"))
      phase2 = set(self.getInt32Vector("OLAP.CNProc.phaseTwoPsets"))
      phase3 = set(self.getInt32Vector("OLAP.CNProc.phaseThreePsets"))

      return len(phase1.intersection(phase3)) == 0 and len(phase2.intersection(phase3)) == 0

    def phaseThreeCoreDisjunct( self ):
      phase12 = set(self.getInt32Vector("OLAP.CNProc.phaseOneTwoCores"))
      phase3 = set(self.getInt32Vector("OLAP.CNProc.phaseThreeCores"))

      return len(phase12.intersection(phase3)) == 0

    def phaseTwoThreePsetEqual( self ):
      phase2 = self.getInt32Vector("OLAP.CNProc.phaseTwoPsets")
      phase3 = self.getInt32Vector("OLAP.CNProc.phaseThreePsets")

      return phase2 == phase3

    def phaseOneTwoThreeCoreEqual( self ):
      phase12 = self.getInt32Vector("OLAP.CNProc.phaseOneTwoCores")
      phase3 = self.getInt32Vector("OLAP.CNProc.phaseThreeCores")

      return phase12 == phase3

    def outputPrefixes( self ):
      return [
        "Observation.DataProducts.Output_Correlated",
        "Observation.DataProducts.Output_Beamformed",
        "Observation.DataProducts.Output_Trigger",
      ]

    def getNrOutputs( self ):
      output_keys = [ "%s.enabled" % (p,) for p in self.outputPrefixes() ]

      return sum( (1 for k in output_keys if k in self and self.getBool(k)) )

    def check( self ):
      """ Check the Parset configuration for inconsistencies. """

      def getBool(k):
        """ A getBool() routine with False as a default value. """
        return k in self and self.getBool(k)

      try:  
        assert self["Observation.nrBeams"] > 0, "No SAPs (beams) specified."
        assert self.getNrOutputs() > 0, "No data output selected."
        assert len(self.stations) > 0, "No stations selected."
        assert len(self.getInt32Vector("Observation.subbandList")) > 0, "No subbands selected."

        # phase 2 and 3 are either disjunct or equal
        assert self.phaseThreePsetDisjunct() or self.phaseTwoThreePsetEqual(), "Phase 2 and 3 should use either disjunct or the same psets."
        assert self.phaseThreeCoreDisjunct() or self.phaseOneTwoThreeCoreEqual(), "Phase 1+2 and 3 should use either disjunct or the same cores."
        assert not (self.phaseThreePsetDisjunct() and self.phaseThreeCoreDisjunct()), "Phase 3 should use either disjunct psets or cores."

        # verify psets used
        nrPsets = len(self.psets)
        for k in [
          "OLAP.CNProc.phaseOnePsets",
          "OLAP.CNProc.phaseTwoPsets",
          "OLAP.CNProc.phaseThreePsets",
        ]:
          psets = self.getInt32Vector( k )
          for p in psets:
            assert p < nrPsets, "Use of pset %d requested in key %s, but only psets [0..%d] are available" % (p,k,nrPsets-1)

        # restrictions on #samples and integration in beam forming modes
        if self.getBool("Observation.DataProducts.Output_Beamformed.enabled"):
          if self["OLAP.CNProc_CoherentStokes.which"] == "XXYY":
            assert int(self["OLAP.CNProc_CoherentStokes.timeIntegrationFactor"]) == 1, "Cannot integrate complex voltages (stokes XXYY) but temporal integration was requested"
            assert int(self["OLAP.CNProc_CoherentStokes.channelsPerSubband"]) in [0, int(self["Observation.channelsPerSubband"])], "Cannot integrate complex voltages (stokes XXYY) but channel collapse was requested"

          # beamforming needs a multiple of 16 samples
          assert int(self["OLAP.CNProc.integrationSteps"]) % 16 == 0, "OLAP.CNProc.integrationSteps should be dividable by 16"

          assert int(self["OLAP.CNProc.integrationSteps"]) % int(self["OLAP.CNProc_CoherentStokes.timeIntegrationFactor"]) == 0, "OLAP.CNProc.integrationSteps should be dividable by OLAP.CNProc_CoherentStokes.timeIntegrationFactor"
          assert int(self["OLAP.CNProc.integrationSteps"]) % int(self["OLAP.CNProc_IncoherentStokes.timeIntegrationFactor"]) == 0, "OLAP.CNProc.integrationSteps should be dividable by OLAP.CNProc_IncoherentStokes.timeIntegrationFactor"
          if not self.phaseThreePsetDisjunct() and not self.phaseThreeCoreDisjunct():
            assert self.getNrBeamFiles() <= len(self.getInt32Vector("Observation.subbandList")), "Cannot create more files than there are subbands."

          # create at least 1 beam
          #assert self.getNrBeams( True ) > 0, "Beam forming requested, but no beams defined. Add at least one beam."

        assert int(self["OLAP.CNProc_CoherentStokes.channelsPerSubband"]) <= int(self["Observation.channelsPerSubband"]), "Coherent Stokes should have the same number or fewer channels than specified for the full observation."
        assert int(self["Observation.channelsPerSubband"]) % int(self["OLAP.CNProc_CoherentStokes.channelsPerSubband"]) == 0, "Coherent Stokes channels should be a whole fraction of the total number of channels."

        assert int(self["OLAP.CNProc_IncoherentStokes.channelsPerSubband"]) <= int(self["Observation.channelsPerSubband"]), "Incoherent Stokes should have the same number or fewer channels than specified for the full observation."
        assert int(self["Observation.channelsPerSubband"]) % int(self["OLAP.CNProc_IncoherentStokes.channelsPerSubband"]) == 0, "Incoherent Stokes channels should be a whole fraction of the total number of channels."

        # verify start/stop times
        assert self["Observation.startTime"] < self["Observation.stopTime"], "Start time (%s) must be before stop time (%s)" % (self["Observation.startTime"],self["Observation.stopTime"])

        # verify stations
        for s in self.stations:
          stationName = s.name.split("_")[0] # remove specific antenna or array name (_hba0 etc) if present
          assert "PIC.Core.%s.phaseCenter" % (stationName,) in self, "Phase center of station '%s' not present in parset." % (stationName,)
      except AssertionError,e:
        error(e);

        self["OLAP.IONProc.parsetError"] = "%s" % (e,)
        return False
      else:
        self["OLAP.IONProc.parsetError"] = ""
        return True

if __name__ == "__main__":
  from optparse import OptionParser,OptionGroup
  import sys

  # parse the command line
  parser = OptionParser( "usage: %prog [options] parset [parset ...]" )

  opgroup = OptionGroup( parser, "Request" )
  opgroup.add_option( "-k", "--key",
                     dest="key",
                     type="string",
                     default="",
                     help="print the given key from the resulting parset" )
  opgroup.add_option( "-P", "--partition",
                     dest="partition",
                     type="string",
                     default=os.environ.get("PARTITION",""),
                     help="use this partition [%default%]" )
  opgroup.add_option( "-r", "--runtime",
                     dest="runtime",
                     type="string",
                     default="",
                     help="starttime,runtime" )
  parser.add_option_group( opgroup )

  # parse arguments
  (options, args) = parser.parse_args()
  
  if not args:
    parser.print_help()
    sys.exit(0)

  parset = Parset()

  for files in args:
    parset.readFile( files )

  if options.partition:
    parset.setPartition( options.partition )

  if options.runtime:
    starttime, runtime = options.runtime.split(",")
    parset.setStartRunTime( starttime, runtime )

  parset.postRead()
  parset.preWrite()
  parset.check()

  if options.key:
    print parset[options.key]
  else:
    # default: print whole parset
    parset.writeFile( "-" )  

  sys.exit(0)

