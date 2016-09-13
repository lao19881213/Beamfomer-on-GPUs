//#  Parset.h: class/struct that holds the Parset information
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: Parset.h 23486 2013-01-10 15:05:57Z mol $

#ifndef LOFAR_INTERFACE_PARSET_H
#define LOFAR_INTERFACE_PARSET_H

// \file
// class/struct that holds the Parset information

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

//# Includes
#include <Common/ParameterSet.h>
#include <Common/LofarBitModeInfo.h>
#include <Common/StreamUtil.h>
#include <Common/StringUtil.h>
#include <Common/LofarLogger.h> 
#include <Interface/BeamCoordinates.h>
#include <Interface/Config.h>
#include <Interface/OutputTypes.h>
#include <Interface/SmartPtr.h>
#include <Stream/Stream.h>
#include <Interface/PrintVector.h>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>
#include <string>

namespace LOFAR {
namespace RTCP {

class Transpose2;
class CN_Transpose2;

enum StokesType { STOKES_I = 0, STOKES_IQUV, STOKES_XXYY, INVALID_STOKES = -1 };
    

// The Parset class is a public struct that can be used as base-class
// for holding Parset related information.
// It can be instantiated with a parset containing Parset information.
class Parset: public ParameterSet
{
  public:
    Parset();
    Parset(const std::string &name);
    Parset(Stream *);

     
    std::string			name() const;
    void			check() const;

    void			write(Stream *) const;

    unsigned			observationID() const;
    double			startTime() const;
    double			stopTime() const;

    unsigned    nrCorrelatedBlocks() const;
    unsigned    nrBeamFormedBlocks() const;

    unsigned			nrStations() const;
    unsigned			nrTabStations() const;
    unsigned			nrMergedStations() const;
    std::vector<std::string>	mergedStationNames() const;
    unsigned			nrBaselines() const;
    unsigned			nrCrossPolarisations() const;
    unsigned			clockSpeed() const; // Hz
    double			subbandBandwidth() const;
    double			sampleDuration() const;
    unsigned			nrBitsPerSample() const;
    size_t			nrBytesPerComplexSample() const;
    std::vector<double>		positions() const;
    std::string			positionType() const;
    std::vector<double>		getRefPhaseCentre() const;
    std::vector<double>		getPhaseCentreOf(const std::string &name) const;
    unsigned			dedispersionFFTsize() const;
    unsigned			CNintegrationSteps() const;
    unsigned			IONintegrationSteps() const;
    unsigned			integrationSteps() const;
    unsigned			coherentStokesTimeIntegrationFactor() const;
    unsigned			coherentStokesNrSubbandsPerFile() const; 
    unsigned			incoherentStokesTimeIntegrationFactor() const;
    unsigned			coherentStokesChannelsPerSubband() const;
    unsigned			incoherentStokesChannelsPerSubband() const;
    unsigned			incoherentStokesNrSubbandsPerFile() const; 
    double			CNintegrationTime() const;
    double			IONintegrationTime() const;
    unsigned			nrSamplesPerChannel() const;
    unsigned			nrSamplesPerSubband() const;
    unsigned			nrSubbandsPerPset() const; 
    unsigned			nrPhase3StreamsPerPset() const; 
    unsigned			nrHistorySamples() const;
    unsigned			nrSamplesToCNProc() const;
    unsigned			inputBufferSize() const; // in samples
    unsigned			maxNetworkDelay() const;
    unsigned			nrPPFTaps() const;
    unsigned			nrChannelsPerSubband() const;
    unsigned			nrCoresPerPset() const;
    std::vector<unsigned>	usedCoresInPset() const;
    std::vector<unsigned>	phaseOneTwoCores() const;
    std::vector<unsigned>	phaseThreeCores() const;
    double			channelWidth() const;
    bool			delayCompensation() const;
    unsigned			nrCalcDelays() const;
    bool			correctClocks() const;
    double			clockCorrectionTime(const std::string &station) const;
    bool			correctBandPass() const;
    bool			hasStorage() const;
    std::string			stationName(int index) const;
    int			        stationIndex(const std::string &name) const;
    std::vector<std::string>	allStationNames() const;
    unsigned			nrPsetsPerStorage() const;
    unsigned			getLofarStManVersion() const;
    std::vector<unsigned>	phaseOnePsets() const;
    std::vector<unsigned>	phaseTwoPsets() const;
    std::vector<unsigned>	phaseThreePsets() const;
    std::vector<unsigned>	usedPsets() const; // union of phasePsets
    unsigned	                totalNrPsets() const; // nr psets in the partition
    bool			phaseThreeDisjunct() const; // if phase 3 does not overlap with phase 1 or 2 in psets or cores
    std::vector<unsigned>	tabList() const;
    bool			conflictingResources(const Parset &otherParset, std::stringstream &error) const;

    int				phaseOnePsetIndex(unsigned pset) const;
    int				phaseTwoPsetIndex(unsigned pset) const;
    int				phaseThreePsetIndex(unsigned pset) const;
    int				phaseOneCoreIndex(unsigned core) const;
    int				phaseTwoCoreIndex(unsigned core) const;
    int				phaseThreeCoreIndex(unsigned core) const;

    std::string			getTransportType(const std::string &prefix) const;

    bool			outputFilteredData() const;
    bool			outputCorrelatedData() const;
    bool			outputBeamFormedData() const;
    bool			outputTrigger() const;
    bool			outputThisType(OutputType) const;

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

    unsigned			nrStreams(OutputType, bool force=false) const;
    unsigned			maxNrStreamsPerPset(OutputType, bool force=false) const;
    static std::string		keyPrefix(OutputType);
    std::string			getHostName(OutputType, unsigned streamNr) const;
    std::string			getFileName(OutputType, unsigned streamNr) const;
    std::string			getDirectoryName(OutputType, unsigned streamNr) const;

    bool			fakeInputData() const;
    bool			checkFakeInputData() const;

    std::string			coherentStokes() const;
    std::string			incoherentStokes() const;
    std::string			bandFilter() const;
    std::string			antennaSet() const;

    size_t          nrCoherentStokes() const { return coherentStokes().size(); }
    size_t          nrIncoherentStokes() const { return incoherentStokes().size(); }

    unsigned			nrBeams() const;
    std::string                 beamTarget(unsigned beam) const;
    double                      beamDuration(unsigned beam) const;

    unsigned			nrTABs(unsigned beam) const;
    std::vector<unsigned>	nrTABs() const;
    unsigned			totalNrTABs() const;
    unsigned			maxNrTABs() const;
    bool                        isCoherent(unsigned beam, unsigned pencil) const;
    BeamCoordinates		TABs(unsigned beam) const;
    double			dispersionMeasure(unsigned beam=0,unsigned pencil=0) const;
    std::vector<std::string>	TABStationList(unsigned beam=0,unsigned pencil=0, bool raw=false) const;

    std::vector<unsigned>	subbandList() const;
    unsigned			nrSubbands() const;
    unsigned			nrSubbandsPerSAP(unsigned sap) const;
    unsigned			nyquistZone() const;

    std::vector<unsigned>	subbandToSAPmapping() const;
    std::vector<double>		subbandToFrequencyMapping() const;
    std::vector<unsigned>	subbandToRSPboardMapping(const std::string &stationName) const;
    std::vector<unsigned>	subbandToRSPslotMapping(const std::string &stationName) const;

    double channel0Frequency( size_t subband ) const;

    unsigned			nrSlotsInFrame() const;
    std::string			partitionName() const;
    bool			realTime() const;
    
    std::vector<double>		getBeamDirection(unsigned beam) const;
    std::string			getBeamDirectionType(unsigned beam) const;

    bool                        haveAnaBeam() const;
    std::vector<double>         getAnaBeamDirection() const;
    std::string                 getAnaBeamDirectionType() const;
          
    struct StationRSPpair {
      std::string station;
      unsigned    rsp;
    };
    
    std::vector<StationRSPpair>	getStationNamesAndRSPboardNumbers(unsigned psetNumber) const;

    std::string			getInputStreamName(const string &stationName, unsigned rspBoardNumber) const;

    std::vector<double>		itsStPositions;

    std::string                 PVSS_TempObsName() const;

    std::string                 AntennaSetsConf() const;
    std::string                 AntennaFieldsDir() const;
    std::string                 HBADeltasDir() const;

    const Transpose2            &transposeLogic() const;
    const CN_Transpose2         &CN_transposeLogic( unsigned pset, unsigned core ) const;

private:
    const std::string		itsName;

    mutable std::string		itsWriteCache;

    mutable SmartPtr<const Transpose2>     itsTransposeLogic;
    mutable SmartPtr<const CN_Transpose2>  itsCN_TransposeLogic;

    void			checkVectorLength(const std::string &key, unsigned expectedSize) const;
    void			checkInputConsistency() const;

    std::vector<double>         getTAB(unsigned beam, unsigned pencil) const;

    void			addPosition(string stName);
    double			getTime(const char *name) const;
    static int			findIndex(unsigned pset, const vector<unsigned> &psets);
    
    std::vector<double>		centroidPos(const string &stations) const;

    bool			compatibleInputSection(const Parset &otherParset, std::stringstream &error) const;
    bool			disjointCores(const Parset &, std::stringstream &error) const;
};

//
// All of the logic for the second transpose.
//

struct StreamInfo {
  unsigned stream;

  unsigned sap;
  unsigned beam;

  bool coherent;
  unsigned nrChannels;     // channels per subband
  unsigned timeIntFactor;  // time integration factor
  unsigned nrStokes;       // total # stokes for this beam
  StokesType stokesType;
  unsigned nrSamples;      // # samples/channel, after temporal integration

  unsigned stokes;
  unsigned part;

  std::vector<unsigned> subbands;

  void log() const;
};

class Transpose2 {
public:
  Transpose2( const Parset &parset );

  unsigned nrStreams() const;

  // compose and decompose a stream number
  unsigned stream( unsigned sap, unsigned beam, unsigned stokes, unsigned part, unsigned startAt = 0) const;
  void decompose( unsigned stream, unsigned &sap, unsigned &beam, unsigned &stokes, unsigned &part ) const;

  std::vector<unsigned> subbands( unsigned stream ) const;
  unsigned nrSubbands( unsigned stream ) const;
  unsigned maxNrSubbands() const;
  unsigned maxNrChannels() const;
  unsigned maxNrSamples() const;

  size_t subbandSize( unsigned stream ) const;

  // the pset/core which processes a certain block of a certain subband
  // note: AsyncTransposeBeams applied the mapping of phaseThreePsets
  unsigned sourceCore( unsigned subband, unsigned block ) const;
  unsigned sourcePset( unsigned subband, unsigned block ) const;

  // the pset/core which processes a certain block of a certain stream
  // note: AsyncTransposeBeams applied the mapping of phaseTwoPsets
  unsigned destCore( unsigned stream, unsigned block ) const;
  unsigned destPset( unsigned stream, unsigned block ) const;

  // if phase2 == phase3, each block in phase3 is processed by more cores (more cores idle to align phases 2 and 3)
  unsigned phaseThreeGroupSize() const;

  const bool phaseThreeDisjunct;

  const unsigned nrChannels;
  const unsigned nrCoherentChannels;
  const unsigned nrIncoherentChannels;
  const unsigned nrSamples;
  const unsigned coherentTimeIntFactor;
  const unsigned incoherentTimeIntFactor;

  const unsigned nrBeams;
  const unsigned coherentNrSubbandsPerFile;
  const unsigned incoherentNrSubbandsPerFile;

  const unsigned nrPhaseTwoPsets;
  const unsigned nrPhaseTwoCores;
  const unsigned nrPhaseThreePsets;
  const unsigned nrPhaseThreeCores;

  const unsigned nrSubbandsPerPset;

  const std::vector<struct StreamInfo> streamInfo;

  const unsigned nrStreamsPerPset;

private:
  std::vector<struct StreamInfo> generateStreamInfo( const Parset &parset ) const;
};

class CN_Transpose2: public Transpose2 {
public:
  CN_Transpose2( const Parset &parset, unsigned myPset, unsigned myCore );

  // the stream to process on (myPset, myCore)
  int myStream( unsigned block ) const;

  // the part number of a subband with an absolute index
  unsigned myPart( unsigned subband, bool coherent ) const;

  const unsigned myPset;
  const unsigned myCore;

  const int phaseTwoPsetIndex;
  const int phaseTwoCoreIndex;
  const int phaseThreePsetIndex;
  const int phaseThreeCoreIndex;
};

} // namespace RTCP
} // namespace LOFAR

#endif
