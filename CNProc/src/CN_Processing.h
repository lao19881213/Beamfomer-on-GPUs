//#  CN_Processing.h: polyphase filter and correlator
//#
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
//#  $Id: CN_Processing.h 23195 2012-12-06 16:01:41Z mol $

#ifndef LOFAR_CNPROC_CN_PROCESSING_H
#define LOFAR_CNPROC_CN_PROCESSING_H

#if 0 || !defined HAVE_BGP
#define C_IMPLEMENTATION
#endif

#include <Interface/Allocator.h>
#include <Interface/BeamFormedData.h>
#include <Interface/Config.h>
#include <Interface/CorrelatedData.h>
#include <Interface/FilteredData.h>
#include <Interface/InputData.h>
#include <Interface/Parset.h>
#include <Interface/SmartPtr.h>
#include <Interface/SubbandMetaData.h>
#include <Interface/TransposedData.h>
#include <Interface/TriggerData.h>

#include <Stream/Stream.h>

#include <AsyncTranspose.h>
#include <AsyncTransposeBeams.h>
#include <BeamFormer.h>
#include <Correlator.h>
#include <Dedispersion.h>
#include <LocationInfo.h>
#include <PPF.h>
#include <PreCorrelationFlagger.h>
#include <PreCorrelationNoChannelsFlagger.h>
#include <PostCorrelationFlagger.h>
#include <Ring.h>
#include <Stokes.h>
#include <Trigger.h>

#include <string>


namespace LOFAR {
namespace RTCP {


class CN_Processing_Base // untemplated helper class
{
  public:
    virtual		~CN_Processing_Base();    

    virtual void	process(unsigned) = 0;
};


template <typename SAMPLE_TYPE> class CN_Processing : public CN_Processing_Base
{
  public:
			CN_Processing(const Parset &, const std::vector<SmartPtr<Stream> > &inputStreams, Stream *(*createStream)(unsigned, const LocationInfo &), const LocationInfo &, Allocator & = heapAllocator, unsigned firstBlock = 0);
			~CN_Processing();

    virtual void	process(unsigned);

  private:
    double		blockAge(); // age of the current block, in seconds since it was observed by the stations
#if defined CLUSTER_SCHEDULING
    void		receiveInput();
#else
    void		transposeInput();
#endif
    int			transposeBeams(unsigned block);
    void		filter();
    void                checkInputForZeros(unsigned station);
    void		dedisperseAfterBeamForming(unsigned beam, double dm);
    void		preCorrelationNoChannelsFlagging();
    void		preCorrelationFlagging();
    void		mergeStations();
    void		formBeams(unsigned sap, unsigned firstBeam, unsigned nrBeams);
    void		receiveBeam(unsigned stream);
    void		correlate();
    void		postCorrelationFlagging();

    void		sendOutput(StreamableData *, Stream *);
    void		finishSendingInput();
    void		finishSendingBeams();

    std::string		itsLogPrefix;
    Allocator           &itsBigAllocator;

    double		itsStartTime, itsIntegrationTime;
    unsigned		itsBlock;

    std::vector<std::string> itsStationNames;
    unsigned		itsNrStations;
    unsigned		itsNrSlotsInFrame;
    unsigned		itsNrSubbands;
    std::vector<unsigned> itsSubbandToSAPmapping;
    std::vector<unsigned> itsNrTABs;
    unsigned		itsMaxNrTABs, itsTotalNrTABs;
    unsigned		itsNrSubbandsPerPset;
    unsigned		itsNrSubbandsPerPart;
    unsigned		itsNrChannels;
    unsigned		itsNrSamplesPerIntegration;
    double		itsCNintegrationTime;
    unsigned		itsPhaseTwoPsetSize, itsPhaseThreePsetSize;
    unsigned		itsPhaseTwoPsetIndex, itsPhaseThreePsetIndex;
    bool		itsPhaseThreeExists, itsPhaseThreeDisjunct;

    const Parset        &itsParset;

    const std::vector<SmartPtr<Stream> > &itsInputStreams;

    SmartPtr<Stream>	itsCorrelatedDataStream;
    SmartPtr<Stream>	itsFinalBeamFormedDataStream;
    SmartPtr<Stream>	itsTriggerDataStream;

    const LocationInfo	&itsLocationInfo;
    const CN_Transpose2 &itsTranspose2Logic;
    std::vector<double> itsCenterFrequencies;
    SmartPtr<Ring>	itsFirstInputSubband, itsCurrentSubband;
    std::vector<double> itsCoherentDMs;
    std::vector<double> itsIncoherentDMs;
    bool		itsFakeInputData;
    bool		itsHasPhaseOne, itsHasPhaseTwo, itsHasPhaseThree;

#if defined HAVE_MPI
    SmartPtr<AsyncTranspose<SAMPLE_TYPE> >	itsAsyncTransposeInput;
    SmartPtr<AsyncTransposeBeams>		itsAsyncTransposeBeams;
#endif

    SmartPtr<InputData<SAMPLE_TYPE> >		itsInputData;
    SmartPtr<SubbandMetaData>			itsInputSubbandMetaData;
    SmartPtr<SubbandMetaData>			itsTransposedSubbandMetaData;
    SmartPtr<TransposedData<SAMPLE_TYPE> > 	itsTransposedInputData;
    SmartPtr<FilteredData>			itsFilteredData;
    SmartPtr<CorrelatedData>			itsCorrelatedData;
    SmartPtr<BeamFormedData>			itsBeamFormedData;
    SmartPtr<TransposedBeamFormedData>		itsTransposedBeamFormedData;
    SmartPtr<FinalBeamFormedData>		itsFinalBeamFormedData;
    SmartPtr<TriggerData>			itsTriggerData;

    std::vector<SmartPtr<PreTransposeBeamFormedData> > itsPreTransposeBeamFormedData;

    struct autoDeallocate { // SmartPtr doesn't work with custom Allocators
      void *ptr;
      Allocator *allocator;

      autoDeallocate(): ptr(0), allocator(0) {}
      ~autoDeallocate() { if (ptr && allocator) allocator->deallocate(ptr); }
    } itsBeamMemory;

    SmartPtr<Arena>                             itsBeamArena;
    SmartPtr<Allocator>                         itsBeamAllocator;

    SmartPtr<PPF<SAMPLE_TYPE> >			itsPPF;
    SmartPtr<BeamFormer>			itsBeamFormer;
    SmartPtr<CoherentStokes>			itsCoherentStokes;
    SmartPtr<IncoherentStokes>			itsIncoherentStokes;
    SmartPtr<Correlator>			itsCorrelator;
    SmartPtr<DedispersionAfterBeamForming>	itsDedispersionAfterBeamForming;
    SmartPtr<DedispersionBeforeBeamForming>	itsDedispersionBeforeBeamForming;
    SmartPtr<PreCorrelationFlagger>		itsPreCorrelationFlagger;
    SmartPtr<PreCorrelationNoChannelsFlagger>	itsPreCorrelationNoChannelsFlagger;
    SmartPtr<PostCorrelationFlagger>		itsPostCorrelationFlagger;
    SmartPtr<Trigger>				itsTrigger;
};

} // namespace RTCP
} // namespace LOFAR

#endif
