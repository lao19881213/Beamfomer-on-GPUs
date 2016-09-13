//#  BeamletBufferToComputeNode.h: Catch RSP ethernet frames and synchronize RSP inputs 
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
//#  $Id: BeamletBufferToComputeNode.h 23195 2012-12-06 16:01:41Z mol $

#ifndef LOFAR_IONPROC_BEAMLET_BUFFER_TO_COMPUTE_NODE_H
#define LOFAR_IONPROC_BEAMLET_BUFFER_TO_COMPUTE_NODE_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

//# Includes
#include <Interface/MultiDimArray.h>
#include <Interface/Parset.h>
#include <Interface/RSPTimeStamp.h>
#include <Interface/SmartPtr.h>
#include <Interface/SubbandMetaData.h>
#include <Stream/Stream.h>
#include <BeamletBuffer.h>
#include <Delays.h>

#include <casa/Quanta/MVDirection.h>

#include <boost/multi_array.hpp>
#include <pthread.h>

#include <string>
#include <vector>


namespace LOFAR {
namespace RTCP {

template <typename SAMPLE_TYPE> class BeamletBufferToComputeNode {
  public:
    BeamletBufferToComputeNode(const Parset &ps, const Matrix<Stream *> &phaseOneTwoStreams, const std::vector<SmartPtr<BeamletBuffer<SAMPLE_TYPE> > > &beamletBuffers, unsigned psetNumber, unsigned firstBlockNumber);
    ~BeamletBufferToComputeNode();
  
    void			 process();

    TimeStamp                    getCurrentTimeStamp() const { return itsCurrentTimeStamp; }
    
  private:
    static void			 limitFlagsLength(SparseSet<unsigned> &flags);

    void			 computeDelays(), computeNextDelays();

    void                         setMetaData( SubbandMetaData &metaData, unsigned psetIndex, unsigned subband );
    void                         sendSubband( Stream *stream, unsigned subband );
    

    void			 startTransaction();
    void			 writeLogMessage() const;
    void			 toComputeNodes();
    void			 stopTransaction();

    std::string                  itsLogPrefix;

    bool			 itsDelayCompensation;
    bool			 itsCorrectClocks;
    bool			 itsNeedDelays;
    bool			 itsIsRealTime;
    std::vector<unsigned>	 itsSubbandToSAPmapping;
    std::vector<unsigned>	 itsSubbandToRSPboardMapping;
    std::vector<unsigned>	 itsSubbandToRSPslotMapping;

    const Matrix<Stream *>       &itsPhaseOneTwoStreams;
    const unsigned               itsNrPhaseOneTwoCoresPerPset;
    
    const Parset		 &itsPS;
    
    TimeStamp			 itsCurrentTimeStamp;
   
    Matrix<double>		 itsDelaysAtBegin;
    Matrix<double>		 itsDelaysAfterEnd;
    Matrix<casa::MVDirection>	 itsBeamDirectionsAtBegin;
    Matrix<casa::MVDirection>	 itsBeamDirectionsAfterEnd;
    unsigned			 itsNrPhaseTwoPsets;
    unsigned			 itsObservationID;
    
    unsigned			 itsMaxNetworkDelay; // in samples
    unsigned                     itsNrSubbands;
    unsigned			 itsNrSubbandsPerPset;
    unsigned			 itsNrSamplesPerSubband;
    unsigned			 itsNrHistorySamples;
    unsigned			 itsNrInputs;
    unsigned			 itsNrBeams;
    unsigned			 itsMaxNrTABs;
    std::vector<unsigned>	 itsNrTABs;

    unsigned			 itsCurrentPhaseOneTwoComputeCore;
    unsigned			 itsPsetNumber;
   
    const std::vector<SmartPtr<BeamletBuffer<SAMPLE_TYPE> > > &itsBeamletBuffers;
    unsigned                     itsBlockNumber;
    SmartPtr<Delays>		 itsDelays;
    double			 itsSubbandBandwidth, itsSampleDuration;
    double			 itsClockCorrectionTime;

    std::vector<TimeStamp>	 itsDelayedStamps;
    std::vector<signed int>	 itsSamplesDelay;
    boost::multi_array<SparseSet<unsigned>, 2> itsFlags;

    Matrix<float>		 itsFineDelaysAtBegin, itsFineDelaysAfterEnd;

    static const unsigned	 itsMaximumDelay = 1000; // samples; roughly 1500 km
    TimeStamp			 itsCorrelationStartTime;
    WallClockTime		 itsWallClock;

    NSTimer			 itsDelayTimer;
};

} // namespace RTCP
} // namespace LOFAR

#endif
