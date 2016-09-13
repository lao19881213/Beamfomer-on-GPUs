//# Parset.h: class/struct that holds the Parset information
//# Copyright (C) 2008-2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
//#
//# This file is part of the LOFAR software suite.
//# The LOFAR software suite is free software: you can redistribute it and/or
//# modify it under the terms of the GNU General Public License as published
//# by the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# The LOFAR software suite is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: Parset.h 27164 2013-10-31 10:30:04Z mol $

#ifndef LOFAR_INTERFACE_PARSET_H
#define LOFAR_INTERFACE_PARSET_H

// \file
// class/struct that holds the Parset information

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <string>
#include <vector>
#include <numeric>
#include <sstream>
#include <algorithm>

#include <Common/ParameterSet.h>
#include <Common/StringUtil.h>
#include <Common/StreamUtil.h>
#include <Stream/Stream.h>
#include <CoInterface/BeamCoordinates.h>
#include <CoInterface/OutputTypes.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/MultiDimArray.h>


namespace LOFAR
{
  namespace Cobalt
  {

    enum StokesType { STOKES_I = 0, STOKES_IQUV, STOKES_XXYY, INVALID_STOKES = -1 };

    // All settings relevant for an observation (well, it should become that,
    // we don't copy all Parset values yet!).
 
    struct ObservationSettings {
      /*
       * Generic information
       */

      // Whether the observation runs at real time. Non-real time
      // observations are not allowed to lose data.
      //
      // key: OLAP.realTime
      bool realTime;

      // The SAS/MAC observation number
      //
      // key: Observation.ObsID
      unsigned observationID;

      // Specified observation start time, in seconds since 1970.
      //
      // key: Observation.startTime
      double startTime;

      // Specified observation stop time, in seconds since 1970.
      //
      // key: Observation.stopTime
      double stopTime;

      // The station clock, in MHz (200 or 160)
      //
      // key: Observation.sampleClock
      unsigned clockMHz;

      // The bandwidth of a single subband, in Hz
      double subbandWidth() const;

      // The number of samples in one block of one subband.
      //
      // key: Cobalt.blockSize
      size_t blockSize;

      // Alias for blockSize
      size_t nrSamplesPerSubband() const;

      // The number of seconds represented by each block.
      double blockDuration() const;

      // The number of bits in each input sample (16, 8, or 4)
      //
      // key: Observation.nrBitsPerSample
      unsigned nrBitsPerSample;

      // The number of polarisations. Set to 2.
      unsigned nrPolarisations;

      // The number of cross polarisations.
      unsigned nrCrossPolarisations() const;

      struct Corrections {
        // Whether the station band pass should be corrected for
        //
        // key: OLAP.correctBandPass
        bool bandPass;

        // Whether the station clock offsets should be corrected for
        //
        // key: OLAP.correctClocks
        bool clock;

        // Whether to dedisperse tied-array beams
        //
        // key: OLAP.coherentDedisperseChannels
        bool dedisperse;
      };
      
      struct Corrections corrections;

      struct DelayCompensation {
        // Whether geometric delays should be compensated for
        //
        // key: OLAP.delayCompensation
        bool enabled;

        // The ITRF position to compensate delays to
        //
        // key: Observation.referencePhaseCenter
        std::vector<double> referencePhaseCenter;
      };
      
      struct DelayCompensation delayCompensation;

      /*
       * Station information
       */

      // The selected antenna set (LBA, HBA_DUAL, HBA_ZERO, etc)
      //
      // key: Observation.antennaSet
      std::string antennaSet;

      // The selected band filter (LBA_30_70, etc)
      //
      // key: Observation.bandFilter
      std::string bandFilter;

      struct Station {
        // The name of the station (CS001LBA, etc)
        //
        // key: OLAP.storageStationNames[stationIdx]
        std::string name;

        // Correction on the station clock, in seconds
        //
        // key: PIC.Core.CS001LBA.clockCorrectionTime
        double clockCorrection;

        // The phase center for which the station beams are corrected, in
        // ITRF [x,y,z].
        //
        // key: PIC.Core.CS001LBA.phaseCenter
        std::vector<double> phaseCenter;

        // The phase correction for this station, in radians.
        //
        // key: PIC.Core.CS001.LBA_INNER.LBA_30_70.phaseCorrection.X
        // key: PIC.Core.CS001.LBA_INNER.LBA_30_70.phaseCorrection.Y
        struct {
          double x;
          double y;
        } phaseCorrection;

        // The delay correction for this station, in seconds
        //
        // key: PIC.Core.CS001.LBA_INNER.LBA_30_70.delayCorrection.X
        // key: PIC.Core.CS001.LBA_INNER.LBA_30_70.delayCorrection.Y
        struct {
          double x;
          double y;
        } delayCorrection;


        // The RSP board to which each subband is mapped
        //
        // key: Observation.Dataslots.CS001LBA.RSPBoardList
        //  or: Observation.rspBoardList
        std::vector<unsigned> rspBoardMap; // [subband]

        // The RSP slot to which each subband is mapped
        //
        // key: Observation.Dataslots.CS001LBA.DataslotList
        //  or: Observation.rspSlotList
        std::vector<unsigned> rspSlotMap;  // [subband]
      };

      // All stations specified as input
      //
      // length: len(OLAP.storageStationNames)
      std::vector<struct Station> stations;

      ssize_t stationIndex(const std::string &name) const;

      /*
       * Resources information:
       *   - what hardware we use (cpus/gpus)
       *   - which nodes receive which stations
       */ 

      struct Node {
        // MPI rank of this node, is the
        // same as the index in the `nodes' vector.
        int rank;

        // Host name
        std::string hostName;

        // CPU number to bind to
        size_t cpu;

        // CUDA GPU numbers to bind to
        std::vector<unsigned> gpus;

        // NIC(s) to bind to (comma seperated)
        //
        // F.e. 'mlx4_0', 'mlx_4_1', 'eth0', etc
        std::string nic;

        // Station indices to forward data for
        std::vector<size_t> stations;
      };

      std::vector<struct Node> nodes;

      /*
       * Pointing information
       */
      struct Direction {
        // Coordinate type (J2000, etc)
        //
        // key: *.directionType
        std::string type;

        // Two angles within the coordinate type (RA/DEC, etc)
        //
        // key: *.angle1
        // key: *.angle2
        double angle1;
        double angle2;
      };

      struct SAP {
        // Direction in which the SAP points
        //
        // key: Observation.Beam[sapIdx].*
        struct Direction direction;

        // Name of target
        //
        // key: Observation.Beam[sapIdx].target
        std::string target;
      };

      // All station beams
      //
      // length: Observation.nrBeams
      std::vector<struct SAP> SAPs;

      struct AnaBeam {
        // Whether the observation employs an analog beam
        //
        // key: Observation.antennaSet starts with "HBA"
        bool enabled;

        // Direction in which the analog beam points
        //
        // key: Observation.AnaBeam[0].*
        struct Direction direction;
      };

      // The analog beam, if any
      struct AnaBeam anaBeam;

      /*
       * Spectral resolution information
       */

      struct Subband {
        // Index (f.e. 0..243)
        //
        // set to: equals the index in the subbands vector
        unsigned idx;

        // Index at station (f.e. 100..343)
        //
        // key: Observation.subbandList[idx]
        unsigned stationIdx;

        // SAP number
        //
        // key: Observation.beamList[idx]
        unsigned SAP;

        // Central frequency (Hz)
        //
        // set to: subbandWidth() * (512 * (nyquistZone() - 1) + stationIdx)
        double centralFrequency;
      };

      // The list of subbands
      //
      // length: len(Observation.subbandList)
      std::vector<struct Subband> subbands;

      size_t nrSubbands(size_t SAP) const;

      struct FileLocation {
        string host;
        string directory;
        string filename;
      };

      /* ===============================
       * Correlator pipeline information
       * ===============================
       */

      struct Correlator {
        // Whether to output correlated data
        //
        // key: Observation.DataProducts.Output_Correlated.enabled
        bool enabled;

        // Number of requested frequency channels per subband
        //
        // key: Observation.channelsPerSubband
        unsigned nrChannels;

        // The bandwidth of a single channel, in Hz
        //
        // set to: subbandWidth() / nrChannels
        double channelWidth;

        // The number of samples in one block of one channel.
        //
        // key: OLAP.CNProc.integrationSteps
        size_t nrSamplesPerChannel;

        // The number of blocks to integrate to obtain the final
        // integration time.
        //
        // key: OLAP.IONProc.integrationSteps
        size_t nrBlocksPerIntegration;

        // The total integration time of all blocks, in seconds.
        double integrationTime() const;

        // The number of blocks in this observation.
        //
        // set to: floor((stopTime - startTime) / integrationTime())
        size_t nrBlocksPerObservation;

        struct Station {
          // The name of this (super)station
          //
          // key: OLAP.tiedArrayStationNames
          std::string name;

          // The list of (input) stations indices to sum for this station
          //
          // key: OLAP.CNProc.tabList
          std::vector<size_t> inputStations;
        };

        // Super-station beam former. NOTE: Not yet supported!! Only
        // here to register how the parset keys work. Until then, this
        // array is the same as the (input) stations array above.
        //
        // The aim is to output this list of stations instead of the
        // ones used for input.
        std::vector<struct Station> stations;

        struct File {
          struct FileLocation location;
        };

        // The list of files to write, indexed by subband
        std::vector<struct File> files; // [subband]
      };

      struct Correlator correlator;

      /* ===============================
       * Beamformer pipeline information
       * ===============================
       */

      struct BeamFormer {
        // Whether beam forming was requested.
        //
        // key: Observation.DataProducts.Output_Beamformed.enabled
        bool enabled;

        struct File {
          size_t streamNr;

          size_t sapNr;
          size_t tabNr;
          size_t stokesNr;
          bool coherent;

          struct FileLocation location;
        };

        // The list of files to write, one file
        // per part/stokes.
        std::vector<struct File> files;

        struct TAB {
          // The direction in wich the TAB points, relative
          // to the SAP's coordinates
          //
          // key: Observation.Beam[sap].TiedArrayBeam[tab].*
          struct Direction directionDelta;

          // Whether the beam is coherent (or incoherent)
          //
          // key: Observation.Beam[sap].TiedArrayBeam[tab].coherent
          bool coherent;

          // The DM with which to dedisperse this beam, or
          // 0.0 for no dedispersion.
          //
          // key: Observation.Beam[sap].TiedArrayBeam[tab].dispersionMeasure
          double dispersionMeasure;

          // The list of files to write, one file
          // per part/stokes.
          std::vector<struct File> files; 
        };

        struct SAP {
          // The TABs to form in this SAP
          //
          // size: Observation.Beam[sap].nrTiedArrayBeams
          std::vector<struct TAB> TABs;

          // Return the number of coherentstokes tabs, 
          size_t nrCoherentTAB() const;

          // Return the number of incoherentstokes tabs
          size_t nrIncoherentTAB() const;

          // calculated at construction time
          size_t nrCoherent;
          size_t nrIncoherent;
        };

        // All SAPs, with information about the TABs to form.
        //
        // size: len(Observation.nrBeams)
        std::vector<struct SAP> SAPs;

        size_t maxNrTABsPerSAP() const;

        struct StokesSettings {
          // Reflection: whether this struct captures
          // coherent or incoherent stokes settings.
          bool coherent;

          // The type of stokes to output
          //
          // key: *.which
          StokesType type;

          // The number of stokes to output
          //
          // set to: nrStokes(type)
          unsigned nrStokes;

          // The requested number of channels
          //
          // key: *.channelsPerSubband
          unsigned nrChannels;

          // The number of samples that need
          // to be integrated temporally, per channel.
          //
          // key: *.timeIntegrationFactor
          size_t timeIntegrationFactor;

          // return the number of samples per channel
          size_t nrSamples(size_t inputBlockSize) const;

          // The number of subbands to store in each file.
          // The last file can have fewer subbands.
          //
          // key: *.subbandsPerFile
          size_t nrSubbandsPerFile;
        };

        // Settings for Coherent Stokes output
        //
        // key: OLAP.CNProc_CoherentStokes.*
        struct StokesSettings coherentSettings;

        // Settings for Incoherent Stokes output
        //
        // key: OLAP.CNProc_IncoherentStokes.*
        struct StokesSettings incoherentSettings;


        // Size of FFT for coherent dedispersion
        //
        // key: OLAP.CNProc.dedispersionFFTsize
        size_t dedispersionFFTsize;
      };

      struct BeamFormer beamFormer;

      // Returns the Nyquist zone number based on bandFilter.
      unsigned nyquistZone() const;

      struct AntennaFieldName {
        std::string station;
        std::string antennaField;

        AntennaFieldName(const std::string &station, const std::string &antennaField): station(station), antennaField(antennaField) {}

        std::string fullName() const {
          return station + antennaField;
        }
      };

      // Constructs the antenna fields ("CS001",HBA0") etc from a set of stations
      // ("CS001","CS002") and the antenna set.
      static std::vector<struct AntennaFieldName> antennaFields(const std::vector<std::string> &stations, const std::string &antennaSet);

      // List of host names to start outputProc on
      std::vector<std::string> outputProcHosts;
    };


    // The Parset class is a public struct that can be used as base-class
    // for holding Parset related information.
    // It can be instantiated with a parset containing Parset information.
    class Parset : public ParameterSet
    {
    public:
      Parset();
      Parset(const std::string &name);
      Parset(Stream *);


      // Transform the parset into an ObservationSettings object
      struct ObservationSettings observationSettings() const;

      // Fill the settings based on the ParameterSet keys.
      // Call this if keys are added or changed.
      void updateSettings();

      struct ObservationSettings settings;

      std::string                 name() const;
      void                        check() const;

      void                        write(Stream *) const;

      unsigned                    observationID() const;
      double                      startTime() const;
      double                      stopTime() const;

      unsigned    nrCorrelatedBlocks() const;
      unsigned    nrBeamFormedBlocks() const;

      unsigned                    nrStations() const;
      unsigned                    nrTabStations() const;
      unsigned                    nrMergedStations() const;
      std::vector<std::string>    mergedStationNames() const;
      unsigned                    nrBaselines() const;
      unsigned                    nrCrossPolarisations() const;
      unsigned                    clockSpeed() const; // Hz
      double                      subbandBandwidth() const;
      double                      sampleDuration() const;
      unsigned                    nrBitsPerSample() const;
      size_t                      nrBytesPerComplexSample() const;
      MultiDimArray<double,2>     positions() const;
      std::string                 positionType() const;
      unsigned                    dedispersionFFTsize() const;
      unsigned                    CNintegrationSteps() const;
      unsigned                    IONintegrationSteps() const;
      unsigned                    integrationSteps() const;
  
      unsigned                    coherentStokesTimeIntegrationFactor() const;
      unsigned                    incoherentStokesTimeIntegrationFactor() const;

      double                      CNintegrationTime() const;
      double                      IONintegrationTime() const;
      unsigned                    nrSamplesPerChannel() const;
      unsigned                    nrSamplesPerSubband() const;
      unsigned                    nrChannelsPerSubband() const;
      double                      channelWidth() const;
      bool                        delayCompensation() const;
      unsigned                    nrCalcDelays() const;
      bool                        correctClocks() const;
      bool                        correctBandPass() const;
      std::string                 stationName(int index) const;
      std::vector<std::string>    allStationNames() const;

      bool                        outputCorrelatedData() const;
      bool                        outputBeamFormedData() const;
      bool                        outputTrigger() const;
      bool outputThisType(OutputType) const;

#if 0
      bool                        onlineFlagging() const;
      bool                        onlinePreCorrelationFlagging() const;
      bool                        onlinePreCorrelationNoChannelsFlagging() const;
      bool                        onlinePostCorrelationFlagging() const;
      bool                        onlinePostCorrelationFlaggingDetectBrokenStations() const;
      unsigned                    onlinePreCorrelationFlaggingIntegration() const;
      std::string                 onlinePreCorrelationFlaggingType(std::string defaultVal) const;
      std::string                 onlinePreCorrelationFlaggingStatisticsType(std::string defaultVal) const;
      std::string                 onlinePostCorrelationFlaggingType(std::string defaultVal) const;
      std::string                 onlinePostCorrelationFlaggingStatisticsType(std::string defaultVal) const;
#endif

      unsigned nrStreams(OutputType, bool force = false) const;
      static std::string keyPrefix(OutputType);
      std::string getHostName(OutputType, unsigned streamNr) const;
      std::string getFileName(OutputType, unsigned streamNr) const;
      std::string getDirectoryName(OutputType, unsigned streamNr) const;

      std::string                 bandFilter() const;
      std::string                 antennaSet() const;

      size_t          nrCoherentStokes() const
      {
        return settings.beamFormer.coherentSettings.nrStokes;
      }
      size_t          nrIncoherentStokes() const
      {
        return settings.beamFormer.incoherentSettings.nrStokes;
      }

      unsigned                    nrBeams() const;
      std::string                 beamTarget(unsigned beam) const;

      unsigned                    nrTABs(unsigned beam) const;
      std::vector<unsigned>       nrTABs() const;
      unsigned                    maxNrTABs() const;
      bool                        isCoherent(unsigned beam, unsigned pencil) const;
      BeamCoordinates             TABs(unsigned beam) const;
      double                      dispersionMeasure(unsigned beam = 0,unsigned pencil = 0) const;
      std::vector<std::string>    TABStationList(unsigned beam = 0,unsigned pencil = 0, bool raw = false) const;

      size_t                      nrSubbands() const;

      double channel0Frequency( size_t subband, size_t nrChannels ) const;

      bool                        realTime() const;

      std::vector<double>         getBeamDirection(unsigned beam) const;
      std::string                 getBeamDirectionType(unsigned beam) const;

      bool                        haveAnaBeam() const;
      std::vector<double>         getAnaBeamDirection() const;
      std::string                 getAnaBeamDirectionType() const;

      std::string                 getInputStreamName(const string &stationName, unsigned rspBoardNumber) const;

      std::vector<double>         itsStPositions;

      std::string                 PVSS_TempObsName() const;

    private:
      const std::string itsName;

      mutable std::string itsWriteCache;

      void                        checkVectorLength(const std::string &key, unsigned expectedSize) const;
      void                        checkInputConsistency() const;

      std::vector<double>         getTAB(unsigned beam, unsigned pencil) const;

      void                        addPosition(string stName);
      double                      getTime(const std::string &name, const std::string &defaultValue) const;

      std::vector<double>         position(const string &name) const;
      std::vector<double>         centroidPos(const string &stations) const;

      struct ObservationSettings::FileLocation         getFileLocation(const std::string outputType, unsigned idx) const;

      // If a parset key is renamed, this function allows the old
      // name to be used as a fall-back.
      //
      // Returns the name of the key in the parset, or `newname' if
      // neither key is defined.
      std::string renamedKey(const std::string &newname, const std::string &oldname) const;
    };
  } // namespace Cobalt
} // namespace LOFAR

#endif

