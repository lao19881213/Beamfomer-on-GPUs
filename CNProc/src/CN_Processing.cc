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
//#  $Id: CN_Processing.cc 24613 2013-04-16 09:07:32Z nieuwpoort $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

//# Includes
#include <CN_Processing.h>
#include <CorrelatorAsm.h>
#include <FIR_Asm.h>
#include <BeamFormer.h>
#include <ContainsOnlyZerosAsm.h>

#include <Common/Timer.h>
#include <Interface/CN_Mapping.h>
#include <Interface/OutputTypes.h>
#include <Interface/PrintVector.h>
#include <Interface/DataFactory.h>
#include <Interface/FakeData.h>
#include <Interface/Align.h>
#include <complex>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

#if defined HAVE_FFTW3
#include <fftw3.h>
#elif defined HAVE_FFTW2
#include <fftw.h>
#else
#error Should have FFTW3 or FFTW2 installed
#endif

#if defined HAVE_BGP
#include <common/bgp_personality_inlines.h>
#include <spi/kernel_interface.h>
#endif

#include <boost/format.hpp>
#include <sys/time.h>

#define SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG 0

#if defined HAVE_BGP
//#define LOG_CONDITION	(itsLocationInfo.rankInPset() == 0)
#define LOG_CONDITION	(itsLocationInfo.rank() == 0)
//#define LOG_CONDITION	1
#else
#define LOG_CONDITION	1
#endif

//#define DEBUG_TRANSPOSE2


// assertion handler for boost
namespace boost {

void assertion_failed(char const * expr, char const * function, char const * file, long line)
{
  THROW(::LOFAR::AssertError, "Assertion failed: " << expr << " in " << function << " (" << file << ":" << line << ")");
}

}


namespace LOFAR {
namespace RTCP {

#if SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG
static  FILE* outputFile;
#endif

static NSTimer computeTimer("computing", true, true);
static NSTimer totalProcessingTimer("global total processing", true, true);


CN_Processing_Base::~CN_Processing_Base()
{
}


template <typename SAMPLE_TYPE> CN_Processing<SAMPLE_TYPE>::CN_Processing(const Parset &parset, const std::vector<SmartPtr<Stream> > &inputStreams, Stream *(*createStream)(unsigned, const LocationInfo &), const LocationInfo &locationInfo, Allocator &bigAllocator, unsigned firstBlock)
:
  itsBigAllocator(bigAllocator),
  itsBlock(firstBlock),
  itsParset(parset),
  itsInputStreams(inputStreams),
  itsLocationInfo(locationInfo),
#if defined HAVE_MPI
  itsTranspose2Logic(parset.CN_transposeLogic(itsLocationInfo.psetNumber(), CN_Mapping::reverseMapCoreOnPset(itsLocationInfo.rankInPset(), itsLocationInfo.psetNumber())))
#else
  itsTranspose2Logic(parset.CN_transposeLogic(0, 0))
#endif
{
#if defined DEBUG_TRANSPOSE2
  if(LOG_CONDITION)
    for (unsigned i = 0; i < itsTranspose2Logic.nrStreams(); i++)
      itsTranspose2Logic.streamInfo[i].log();
#endif

#if defined HAVE_MPI
  unsigned myPset	    = itsLocationInfo.psetNumber();
  unsigned myCoreInPset	    = CN_Mapping::reverseMapCoreOnPset(itsLocationInfo.rankInPset(), myPset);
#else
  unsigned myPset	    = 0;
  unsigned myCoreInPset	    = 0;
#endif

  itsStartTime = parset.startTime();
  itsIntegrationTime = parset.CNintegrationTime();

  std::vector<unsigned> phaseOneTwoCores = parset.phaseOneTwoCores();
  std::vector<unsigned> phaseThreeCores = parset.phaseThreeCores();

  std::vector<unsigned> phaseOnePsets = parset.phaseOnePsets();
  std::vector<unsigned> phaseTwoPsets = parset.phaseTwoPsets();
  std::vector<unsigned> phaseThreePsets = parset.phaseThreePsets();

#if defined CLUSTER_SCHEDULING
#define itsHasPhaseOne false
#else
  itsHasPhaseOne             = parset.phaseOnePsetIndex(myPset) >= 0   && parset.phaseOneCoreIndex(myCoreInPset) >= 0;
#endif
  itsHasPhaseTwo             = parset.phaseTwoPsetIndex(myPset) >= 0   && parset.phaseTwoCoreIndex(myCoreInPset) >= 0;
  itsHasPhaseThree           = parset.phaseThreePsetIndex(myPset) >= 0 && parset.phaseThreeCoreIndex(myCoreInPset) >= 0;

  itsPhaseTwoPsetIndex       = itsHasPhaseTwo ? parset.phaseTwoPsetIndex( myPset ) : 0;
  itsPhaseThreePsetIndex     = itsHasPhaseThree ? parset.phaseThreePsetIndex( myPset ) : 0;

  itsPhaseTwoPsetSize        = phaseTwoPsets.size();
  itsPhaseThreePsetSize      = phaseThreePsets.size();

  itsPhaseThreeExists	     = parset.outputBeamFormedData() || parset.outputTrigger();
  itsPhaseThreeDisjunct      = parset.phaseThreeDisjunct();

  itsLogPrefix = boost::str(boost::format("[obs %u phases %d%d%d] ") % parset.observationID() % (itsHasPhaseOne ? 1 : 0) % (itsHasPhaseTwo ? 1 : 0) % (itsHasPhaseThree ? 1 : 0));

  if (LOG_CONDITION)
    LOG_INFO_STR(itsLogPrefix << "----- Observation start");

  itsStationNames            = parset.allStationNames();
  itsNrStations	             = parset.nrStations();
  unsigned nrMergedStations  = parset.nrMergedStations();
  itsNrSubbands              = parset.nrSubbands();
  itsSubbandToSAPmapping     = parset.subbandToSAPmapping();
  itsNrTABs                  = parset.nrTABs();
  itsMaxNrTABs	             = parset.maxNrTABs();
  itsTotalNrTABs	     = parset.totalNrTABs();
  itsNrSubbandsPerPset       = parset.nrSubbandsPerPset();
  itsCenterFrequencies       = parset.subbandToFrequencyMapping();
  itsNrChannels		     = parset.nrChannelsPerSubband();
  itsNrSamplesPerIntegration = parset.CNintegrationSteps();
  itsFakeInputData           = parset.fakeInputData();
  itsNrSlotsInFrame          = parset.nrSlotsInFrame();
  itsCNintegrationTime       = parset.CNintegrationTime();


#if SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG 
  stringstream filename;
  filename << "/var/scratch/rob/" << myPset << "." << myCoreInPset << ".myFilteredData";
  outputFile = fopen(filename.str().c_str(), "w");
  fwrite(&itsNrStations, sizeof(unsigned), 1, outputFile);
  fwrite(&itsNrSubbands, sizeof(unsigned), 1, outputFile);
  fwrite(&itsNrChannels, sizeof(unsigned), 1, outputFile);
  unsigned tmp = NR_POLARIZATIONS;
  fwrite(&tmp, sizeof(unsigned), 1, outputFile);
  fflush(outputFile);
#endif // SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG


  if (itsFakeInputData && LOG_CONDITION)
    LOG_WARN_STR(itsLogPrefix << "Generating fake input data -- any real input is discarded!");

  // my index in the set of cores which can be used
  unsigned phaseTwoCoreIndex  = parset.phaseTwoCoreIndex( myCoreInPset );

  if (itsHasPhaseOne) {
    itsFirstInputSubband = new Ring(0, itsNrSubbandsPerPset, phaseTwoCoreIndex, phaseOneTwoCores.size());
    itsInputData = new InputData<SAMPLE_TYPE>(itsPhaseTwoPsetSize, parset.nrSamplesToCNProc(), itsBigAllocator);
    itsInputSubbandMetaData = new SubbandMetaData(itsPhaseTwoPsetSize, itsMaxNrTABs + 1);

    // skip ahead to the first block
    itsFirstInputSubband->skipFirstBlocks(itsBlock);
  }

  if (itsHasPhaseTwo || itsHasPhaseThree)
    itsBeamFormer = new BeamFormer(parset);

  if (itsHasPhaseTwo) {
    itsCurrentSubband = new Ring(itsPhaseTwoPsetIndex, itsNrSubbandsPerPset, phaseTwoCoreIndex, phaseOneTwoCores.size());

    // skip ahead to the first block
    itsCurrentSubband->skipFirstBlocks(itsBlock);

    itsTransposedSubbandMetaData = new SubbandMetaData(itsNrStations, itsTotalNrTABs + 1);
    itsTransposedInputData = new TransposedData<SAMPLE_TYPE>(itsNrStations, parset.nrSamplesToCNProc(), itsBigAllocator);

#if defined HAVE_MPI
    if (LOG_CONDITION)
      LOG_DEBUG_STR("Processes subbands " << itsCurrentSubband->list());
#endif // HAVE_MPI

    itsPPF	    = new PPF<SAMPLE_TYPE>(itsNrStations, itsNrChannels, itsNrSamplesPerIntegration, parset.subbandBandwidth() / itsNrChannels, parset.delayCompensation() || itsTotalNrTABs > 1 || parset.correctClocks(), parset.correctBandPass(), itsLocationInfo.rank() == 0);
    itsFilteredData = new FilteredData(parset.nrStations(), parset.nrChannelsPerSubband(), parset.CNintegrationSteps(), itsBigAllocator);

    if (parset.onlineFlagging() && parset.onlinePreCorrelationFlagging()) {
      itsPreCorrelationFlagger = new PreCorrelationFlagger(parset, itsNrStations, itsNrSubbands, itsNrChannels, itsNrSamplesPerIntegration);
      if (LOG_CONDITION)
        LOG_DEBUG_STR("Online PreCorrelation flagger enabled");
    } else {
      itsPreCorrelationFlagger = NULL;
    }

    if (parset.onlineFlagging() && parset.onlinePreCorrelationNoChannelsFlagging()) {
      itsPreCorrelationNoChannelsFlagger = new PreCorrelationNoChannelsFlagger(parset, myPset, myCoreInPset, parset.correctBandPass(), itsNrStations, itsNrSubbands, itsNrChannels, itsNrSamplesPerIntegration);
      if (LOG_CONDITION)
        LOG_DEBUG_STR("Online PreCorrelation no channels flagger enabled");
    } else {
      itsPreCorrelationNoChannelsFlagger = NULL;
    }

    if (parset.outputCorrelatedData()) {
      itsCorrelator	      = new Correlator(itsBeamFormer->getStationMapping(), itsNrChannels, itsNrSamplesPerIntegration);
      itsCorrelatedData       = (CorrelatedData*)newStreamableData(parset, CORRELATED_DATA, -1, itsBigAllocator);
      itsCorrelatedDataStream = createStream(CORRELATED_DATA, itsLocationInfo);
    }  

    if (parset.onlineFlagging() && parset.onlinePostCorrelationFlagging()) {
      itsPostCorrelationFlagger = new PostCorrelationFlagger(parset, nrMergedStations, itsNrSubbands, itsNrChannels);
      if (LOG_CONDITION)
        LOG_DEBUG_STR("Online PostCorrelation flagger enabled");
    } else {
      itsPostCorrelationFlagger = NULL;
    }


    if (parset.onlineFlagging() && parset.onlinePostCorrelationFlagging() && parset.onlinePostCorrelationFlaggingDetectBrokenStations()) {
      if (LOG_CONDITION)
        LOG_DEBUG_STR("Online PostCorrelation flagger Detect Broken Stations enabled");
    }
  }

    if (parset.outputBeamFormedData() || parset.outputTrigger()) {
      itsBeamFormedData = new BeamFormedData(BeamFormer::BEST_NRBEAMS, itsNrChannels, itsNrSamplesPerIntegration, itsBigAllocator);

      if (LOG_CONDITION)
        LOG_DEBUG_STR("Considering dedispersion for " << itsTotalNrTABs << " pencil beams");

      itsCoherentDMs.resize(itsTotalNrTABs, 0.0);
      itsIncoherentDMs.resize(itsTotalNrTABs, 0.0);

      bool dedisperseCoherent = false;
      bool dedisperseIncoherent = false;
      unsigned i = 0;
      unsigned nrSAPs = parset.nrBeams();

      for (unsigned sap = 0; sap < nrSAPs; sap++) {
        for (unsigned pencil = 0; pencil < itsNrTABs[sap]; pencil++) {
          double DM = parset.dispersionMeasure(sap, pencil);
          if(LOG_CONDITION) LOG_DEBUG_STR("DM for beam " << sap << " TAB " << pencil << " is " << DM);

          if (DM != 0.0) {
            if (parset.isCoherent(sap, pencil)) {
              dedisperseCoherent = true;
              itsCoherentDMs[i] = DM;
            } else {
              dedisperseIncoherent = true;
              itsIncoherentDMs[i] = DM;
            }
          }

          i++;
        }
      }

      if (dedisperseCoherent) {
        if(LOG_CONDITION) LOG_DEBUG("Doing dedispersion for coherent data");
        itsDedispersionAfterBeamForming = new DedispersionAfterBeamForming(parset, itsBeamFormedData, itsCurrentSubband->list(), itsCoherentDMs);
      } else {
        if(LOG_CONDITION) LOG_DEBUG("NOT doing dedispersion for coherent data");
      }

      if (dedisperseIncoherent) {
        if(LOG_CONDITION) LOG_DEBUG("Doing dedispersion for incoherent data");
        itsDedispersionBeforeBeamForming = new DedispersionBeforeBeamForming(parset, itsFilteredData, itsCurrentSubband->list(), itsIncoherentDMs);
      } else {
        if(LOG_CONDITION) LOG_DEBUG("NOT doing dedispersion for incoherent data");
      }

      // Our assembly code (BeamFormerAsm) requires groups of beams it processes to
      // be consecutive, so store everything in one big block, controlling the offsets.

      // determine total memory required to process one subband in each SAP
      vector<size_t> totalsizes(parset.nrBeams(), 0);
      for (unsigned i = 0; i < itsTranspose2Logic.nrStreams(); i++) {
        const StreamInfo &info = itsTranspose2Logic.streamInfo[i];

        // ignore multiple parts since we'll always only process one subband, and thus one part
        if (info.part != 0)
          continue;

        totalsizes[info.sap] += align(itsTranspose2Logic.subbandSize(i), StreamableData::alignment);
      }

      // allocate memory for the largest SAP
      size_t max_totalsize = *std::max_element(totalsizes.begin(), totalsizes.end());

      itsBeamMemory.allocator = &itsBigAllocator;
      itsBeamMemory.ptr       = itsBigAllocator.allocate(max_totalsize, StreamableData::alignment);
      itsBeamArena      = new FixedArena(itsBeamMemory.ptr, max_totalsize);
      itsBeamAllocator  = new SparseSetAllocator(*itsBeamArena.get()); // allocates consecutively

      itsPreTransposeBeamFormedData.resize(itsMaxNrTABs);

      if (LOG_CONDITION) {
        LOG_DEBUG_STR("MaxNrTABs = " << itsMaxNrTABs << ", TotalNrTABs = " << itsTotalNrTABs);
        LOG_DEBUG_STR("Allocated " << max_totalsize << " bytes for beam forming.");
      }
  }

  if (itsHasPhaseTwo || itsHasPhaseThree) {
    itsCoherentStokes   = new CoherentStokes(itsNrChannels, itsNrSamplesPerIntegration);
    itsIncoherentStokes = new IncoherentStokes(itsNrChannels, itsNrSamplesPerIntegration, nrMergedStations, parset.nrChannelsPerSubband() / parset.incoherentStokesChannelsPerSubband(), itsDedispersionBeforeBeamForming, itsBigAllocator);
  }

#if defined HAVE_MPI
  if (itsHasPhaseOne || itsHasPhaseTwo)
    itsAsyncTransposeInput = new AsyncTranspose<SAMPLE_TYPE>(itsHasPhaseOne, itsHasPhaseTwo, myCoreInPset, itsLocationInfo, phaseOnePsets, phaseTwoPsets);

  if (itsPhaseThreeExists && (itsHasPhaseTwo || itsHasPhaseThree))
    itsAsyncTransposeBeams = new AsyncTransposeBeams(itsHasPhaseTwo, itsHasPhaseThree, itsNrSubbands, itsLocationInfo, phaseTwoPsets, phaseOneTwoCores, phaseThreePsets, phaseThreeCores);
#endif // HAVE_MPI

  if (itsHasPhaseThree) {
    if (parset.outputBeamFormedData() || parset.outputTrigger()) {
      itsTransposedBeamFormedData  = new TransposedBeamFormedData(itsTranspose2Logic.maxNrSubbands(), itsTranspose2Logic.maxNrChannels(), itsTranspose2Logic.maxNrSamples(), itsBigAllocator);
      itsFinalBeamFormedData	   = (FinalBeamFormedData*)newStreamableData(parset, BEAM_FORMED_DATA, -1, itsBigAllocator);
      itsFinalBeamFormedDataStream = createStream(BEAM_FORMED_DATA, itsLocationInfo);
    }

    if (parset.outputTrigger()) {
      itsTrigger	   = new Trigger;
      itsTriggerData	   = (TriggerData*)newStreamableData(parset, TRIGGER_DATA, -1, itsBigAllocator);
      itsTriggerDataStream = createStream(TRIGGER_DATA, itsLocationInfo);
    }
  }
}


template <typename SAMPLE_TYPE> CN_Processing<SAMPLE_TYPE>::~CN_Processing()
{
  if (LOG_CONDITION)
    LOG_INFO_STR(itsLogPrefix << "----- Observation finished");

  // destruct all uses of itsBeamMemory so it can be freed properly
  itsPreTransposeBeamFormedData.resize(0);
  itsBeamAllocator = 0;
  itsBeamArena = 0;

  // don't accumulate plans in memory, as we might run out or create fragmentation
#if defined HAVE_FFTW3
  fftwf_forget_wisdom();
  fftwf_cleanup();  
#elif defined HAVE_FFTW2
  fftw_forget_wisdom();
#endif  

#if SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG
  fclose(outputFile);
#endif
}


template <typename SAMPLE_TYPE> double CN_Processing<SAMPLE_TYPE>::blockAge()
{
  struct timeval tv;
  double observeTime = itsStartTime + itsBlock * itsIntegrationTime;
  double now;

  gettimeofday(&tv,0);
  now = 1.0*tv.tv_sec + 1.0*tv.tv_usec/1000000.0;

  return now - observeTime;
}


#if defined CLUSTER_SCHEDULING

template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::receiveInput()
{
  SubbandMetaData metaData(1, itsMaxNrTABs + 1);

  for (unsigned stat = 0; stat < itsNrStations; stat ++) {
    if (LOG_CONDITION)
      LOG_DEBUG_STR(itsLogPrefix << "Receiving input of station " << stat);

    // receive meta data
    metaData.read(itsInputStreams[stat]); // FIXME
    memcpy(&itsTransposedSubbandMetaData->subbandInfo(stat), &metaData.subbandInfo(0), metaData.itsSubbandInfoSize);

    // receive samples
    itsInputStreams[stat]->read(itsTransposedInputData->samples[stat].origin(), itsTransposedInputData->samples[stat].num_elements() * sizeof(SAMPLE_TYPE));
  }
}

#else

template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::transposeInput()
{
#if defined HAVE_MPI
  if (itsHasPhaseOne)
    itsInputSubbandMetaData->read(itsInputStreams[0]); // sync read the meta data

  if (itsHasPhaseTwo && *itsCurrentSubband < itsNrSubbands) {
    NSTimer postAsyncReceives("post async receives", LOG_CONDITION, true);
    postAsyncReceives.start();
    itsAsyncTransposeInput->postAllReceives(itsTransposedSubbandMetaData, itsTransposedInputData);
    postAsyncReceives.stop();
  }

  // We must not try to read data from I/O node if our subband does not exist.
  // Also, we cannot do the async sends in that case.
  if (itsHasPhaseOne) { 
    static NSTimer readTimer("receive timer", true, true);
    static NSTimer phaseOneTimer("phase one timer", true, true);

    phaseOneTimer.start();

    if (LOG_CONDITION)
      LOG_DEBUG_STR(itsLogPrefix << "Start reading at t = " << blockAge());
    
    NSTimer asyncSendTimer("async send", LOG_CONDITION, true);

    unsigned subband = *itsFirstInputSubband;
    itsFirstInputSubband->next();

    for (unsigned i = 0; i < itsPhaseTwoPsetSize; i ++, subband += itsNrSubbandsPerPset) {
      //unsigned subband = (*itsCurrentSubband % itsNrSubbandsPerPset) + (i * itsNrSubbandsPerPset);

      if (subband < itsNrSubbands) {
	readTimer.start();
	itsInputData->readOne(itsInputStreams[0], i); // Synchronously read 1 subband from my IO node.
	readTimer.stop();
	asyncSendTimer.start();

	itsAsyncTransposeInput->asyncSend(i, itsInputSubbandMetaData, itsInputData); // Asynchronously send one subband to another pset.
	asyncSendTimer.stop();
      }
    }

    phaseOneTimer.stop();
  }
#else // ! HAVE_MPI
  if (itsHasPhaseOne) {
    static NSTimer readTimer("receive timer", true, true);
    readTimer.start();
    itsInputSubbandMetaData->read(itsInputStreams[0]);
    itsInputData->read(itsInputStreams[0], false);
    readTimer.stop();
  }
#endif // HAVE_MPI
}

#endif


template <typename SAMPLE_TYPE> int CN_Processing<SAMPLE_TYPE>::transposeBeams(unsigned block)
{
  int myStream          = itsTranspose2Logic.myStream( block );
  bool streamToProcess  = itsHasPhaseThree && myStream >= 0;

  if (!streamToProcess) {
    // check whether we really have nothing to process
    unsigned myPset = itsTranspose2Logic.phaseThreePsetIndex;
    unsigned myCore = itsTranspose2Logic.phaseThreeCoreIndex;

    for (unsigned s = 0; s < itsTranspose2Logic.nrStreams(); s++) {
      ASSERTSTR(!(myPset == itsTranspose2Logic.destPset(s, block) && myCore == itsTranspose2Logic.destCore(s, block)),
       "I'm (" << myPset << ", " << myCore << ") and should process stream " << s << " for block " << block << " but myStream( ) does not return it.");
    }
  }

  if (itsHasPhaseTwo && *itsCurrentSubband < itsNrSubbands) {
    unsigned subband = *itsCurrentSubband;

    ASSERTSTR((unsigned)itsTranspose2Logic.phaseTwoPsetIndex == itsTranspose2Logic.sourcePset( subband, block ) && (unsigned)itsTranspose2Logic.phaseTwoCoreIndex == itsTranspose2Logic.sourceCore( subband, block ),
     "I'm (" << itsTranspose2Logic.phaseTwoPsetIndex << ", " << itsTranspose2Logic.phaseTwoCoreIndex << ") . For block " << block << ", I have subband " << subband << ", but the logic expects that subband from (" << itsTranspose2Logic.sourcePset( subband, block ) << ", " << itsTranspose2Logic.sourceCore( subband, block ) << ")" );
  }

#if defined HAVE_MPI
  if (streamToProcess) {
    ASSERTSTR((unsigned)itsTranspose2Logic.phaseThreePsetIndex == itsTranspose2Logic.destPset( myStream, block ) && (unsigned)itsTranspose2Logic.phaseThreeCoreIndex == itsTranspose2Logic.destCore( myStream, block ),
     "I'm (" << itsTranspose2Logic.phaseThreePsetIndex << ", " << itsTranspose2Logic.phaseThreeCoreIndex << ") . According to the logic, for block " << block << ", I'm to handle stream " << myStream << ", yet that stream is to be handled by (" << itsTranspose2Logic.destPset( myStream, block ) << ", " << itsTranspose2Logic.destCore( myStream, block ) << ")" );

    if (LOG_CONDITION)
      LOG_DEBUG_STR(itsLogPrefix << "Phase 3 starting at t = " << blockAge());

    const StreamInfo &info = itsTranspose2Logic.streamInfo[myStream];

    itsTransposedBeamFormedData->setDimensions(info.subbands.size(), info.nrChannels, info.nrSamples);

    if (itsFinalBeamFormedData != 0) {
      itsFinalBeamFormedData->setDimensions(info.nrSamples, info.subbands.size(), info.nrChannels);
    }  

    static NSTimer postAsyncReceives("post async beam receives", true, true);
    postAsyncReceives.start();

    for (unsigned sb = 0; sb < info.subbands.size(); sb++) {
      unsigned subband = info.subbands[sb];

      unsigned pset = itsTranspose2Logic.sourcePset( subband, block );
      unsigned core = itsTranspose2Logic.sourceCore( subband, block );

#ifdef DEBUG_TRANSPOSE2      
      LOG_DEBUG_STR(itsLogPrefix << "transpose: (stream, subband, block) <- (pset, core): (" << myStream << ", " << subband << ", " << block << ") <- (" << pset << ", " << core << ")" );
#endif        
      itsAsyncTransposeBeams->postReceive(itsTransposedBeamFormedData.get(), sb, subband, myStream, pset, core);
    }

    postAsyncReceives.stop();
  }

  if (itsHasPhaseTwo && *itsCurrentSubband < itsNrSubbands) {
    if (LOG_CONDITION)
      LOG_DEBUG_STR(itsLogPrefix << "Start sending beams at t = " << blockAge());

    static NSTimer asyncSendTimer("async beam send", true, true);

    /* overlap computation and transpose */
    /* this makes async send timing worse -- due to caches? remember that we do
       async sends, so we're not actually using the data we just calculated, just
       references to it.
       
       overlapping computation and transpose does improve the latency though, so
       it might still be worthwhile if the increase in cost is acceptable. */

    // retrieve info about which beams and parts our subband will end up in
    unsigned subband = *itsCurrentSubband;
    unsigned sap = itsSubbandToSAPmapping[subband];

    unsigned nrBeams = itsNrTABs[sap];
    unsigned coherentPart   = itsTranspose2Logic.myPart(subband, true);
    unsigned incoherentPart = itsTranspose2Logic.myPart(subband, false);

    //LOG_DEBUG_STR("I process subband " << subband << " which belongs to sap " << sap << " part " << part);

    unsigned stream = 0;
    
    // form and send beams for this SAP
    for (unsigned beam = 0; beam < nrBeams;) { // beam is incremented in inner for-loop
      unsigned groupSize;

      // go to part 0 first, to determine coherency (which determines the part #)
      stream = itsTranspose2Logic.stream(sap, beam, 0, 0, stream);
      const StreamInfo &info = itsTranspose2Logic.streamInfo[stream];
      const unsigned part = info.coherent ? coherentPart : incoherentPart;

      if (info.coherent) {
        // a coherent beam -- look BEST_NRBEAMS ahead to see if we can process them at the same time

        groupSize = std::min(nrBeams - beam, +BeamFormer::BEST_NRBEAMS); // unary + to avoid requiring a reference
        unsigned stream2 = stream;

        // determine how many beams (up to groupSize) are coherent
        for (unsigned i = 1; i < groupSize; i++ ) {
          stream2 = itsTranspose2Logic.stream(sap, beam+i, 0, 0, stream2);
          const StreamInfo &info2 = itsTranspose2Logic.streamInfo[stream2];

          if (!info2.coherent) {
            groupSize = i;
            break;
          }
        }

        if(LOG_CONDITION)
          LOG_DEBUG_STR(itsLogPrefix << "Forming beams " << beam << " .. " << (beam+groupSize-1) << " at t = " << blockAge());
        formBeams(sap, beam, groupSize);
      } else {
        groupSize = 1;
      }

      for (unsigned i = 0; i < groupSize; i ++, beam ++) {
        stream = itsTranspose2Logic.stream(sap, beam, 0, part, stream);

        const StreamInfo &info = itsTranspose2Logic.streamInfo[stream];

        ASSERT( beam < itsPreTransposeBeamFormedData.size() );
        ASSERT( itsPreTransposeBeamFormedData[beam].get() == 0 );

        itsPreTransposeBeamFormedData[beam] = new PreTransposeBeamFormedData(info.nrStokes, info.nrChannels, info.nrSamples, *itsBeamAllocator.get());

        ASSERT( itsPreTransposeBeamFormedData[beam].get() != NULL );

        if (info.coherent) {
          if (itsDedispersionAfterBeamForming != 0) {
            if(LOG_CONDITION)
              LOG_DEBUG_STR(itsLogPrefix << "Dedispersing beam-formed data at t = " << blockAge());

            dedisperseAfterBeamForming(i, itsCoherentDMs[beam]);
          }

          switch (info.stokesType) {
            case STOKES_I:
              if(LOG_CONDITION)
                LOG_DEBUG_STR(itsLogPrefix << "Calculating coherent Stokes I at t = " << blockAge());
              itsCoherentStokes->calculate<false>(itsBeamFormedData.get(), itsPreTransposeBeamFormedData[beam].get(), i, info);
              break;

            case STOKES_IQUV:
              if(LOG_CONDITION)
                LOG_DEBUG_STR(itsLogPrefix << "Calculating coherent Stokes IQUV at t = " << blockAge());
              itsCoherentStokes->calculate<true>(itsBeamFormedData.get(), itsPreTransposeBeamFormedData[beam].get(), i, info);
              break;

            case STOKES_XXYY:
              if(LOG_CONDITION)
                LOG_DEBUG_STR(itsLogPrefix << "Calculating coherent Stokes XXYY at t = " << blockAge());
              itsBeamFormer->preTransposeBeam(itsBeamFormedData.get(), itsPreTransposeBeamFormedData[beam].get(), i);
              break;

            case INVALID_STOKES:
              ASSERT( false );
              break;
          }
        } else {  
          // TODO: optimise dedispersion to only do the forwardFFT once

          switch (info.stokesType) {
            case STOKES_I:
              if(LOG_CONDITION)
                LOG_DEBUG_STR(itsLogPrefix << "Calculating incoherent Stokes I at t = " << blockAge());
              itsIncoherentStokes->calculate<false>(itsFilteredData.get(), itsPreTransposeBeamFormedData[beam].get(), itsBeamFormer->getStationMapping(), info, subband, itsIncoherentDMs[beam]);
              break;

            case STOKES_IQUV:
              if(LOG_CONDITION)
                LOG_DEBUG_STR(itsLogPrefix << "Calculating incoherent Stokes IQUV at t = " << blockAge());
              itsIncoherentStokes->calculate<true>(itsFilteredData.get(), itsPreTransposeBeamFormedData[beam].get(), itsBeamFormer->getStationMapping(), info, subband, itsIncoherentDMs[beam]);
              break;

            case STOKES_XXYY:
              ASSERT( false );
              break;

            case INVALID_STOKES:
              ASSERT( false );
              break;
          }
        }

        if(LOG_CONDITION)
          LOG_DEBUG_STR(itsLogPrefix << "Done calculating Stokes at t = " << blockAge());

        asyncSendTimer.start();

        for (unsigned stokes = 0; stokes < info.nrStokes; stokes ++) {
          // calculate which (pset,core) needs the beam part
          stream = itsTranspose2Logic.stream(sap, beam, stokes, part, stream);

          unsigned pset = itsTranspose2Logic.destPset(stream, block);
          unsigned core = itsTranspose2Logic.destCore(stream, block);

#ifdef DEBUG_TRANSPOSE2      
          LOG_DEBUG_STR(itsLogPrefix << "transpose: (stream, subband, block) -> (pset, core): (" << stream << ", " << *itsCurrentSubband << ", " << block << ") -> (" << pset << ", " << core << ")" );
#endif
          itsAsyncTransposeBeams->asyncSend(pset, core, *itsCurrentSubband, stokes, stream, itsPreTransposeBeamFormedData[beam].get()); // Asynchronously send one beam to another pset.
        }

        asyncSendTimer.stop();
      }
    }  
  }
#endif // HAVE_MPI

  return streamToProcess ? myStream : -1;
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::filter()
{
#if defined HAVE_MPI && !defined CLUSTER_SCHEDULING
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start filtering at t = " << blockAge());

  NSTimer asyncReceiveTimer("wait for any async receive", LOG_CONDITION, true);
  static NSTimer timer("filter timer", true, true);

  timer.start();

  for (unsigned i = 0; i < itsNrStations; i ++) {
    asyncReceiveTimer.start();
    unsigned stat = itsAsyncTransposeInput->waitForAnyReceive();
    asyncReceiveTimer.stop();

    checkInputForZeros(stat);

    computeTimer.start();
    itsPPF->doWork(stat, itsCenterFrequencies[*itsCurrentSubband], itsTransposedSubbandMetaData, itsTransposedInputData, itsFilteredData);
    computeTimer.stop();
  }

  timer.stop();
#else
  for (unsigned stat = 0; stat < itsNrStations; stat ++) {
    computeTimer.start();
    itsPPF->doWork(stat, itsCenterFrequencies[*itsCurrentSubband], itsTransposedSubbandMetaData, itsTransposedInputData, itsFilteredData);
    computeTimer.stop();
  }
#endif

  if (itsFakeInputData)
    FakeData(itsParset).fill(itsFilteredData, *itsCurrentSubband);
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::checkInputForZeros(unsigned station)
{
#ifdef HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start checking for zeroes at " << MPI_Wtime());
#endif

  static NSTimer timer("input zero-check timer", true, true);

  timer.start();

  const unsigned nrSamplesToCNProc = itsNrSamplesPerIntegration * itsNrChannels;

  const SparseSet<unsigned> &flags = itsTransposedSubbandMetaData->getFlags(station);
  SparseSet<unsigned> validSamples = flags.invert(0, nrSamplesToCNProc);

  bool allzeros = true;

  // only consider non-flagged samples, as flagged samples aren't necessarily zero
  for (SparseSet<unsigned>::const_iterator it = validSamples.getRanges().begin(); allzeros && it != validSamples.getRanges().end(); ++it) {

#ifdef HAVE_BGP
    unsigned first = it->begin;
    unsigned nrSamples = it->end - it->begin;

    ASSERT(NR_POLARIZATIONS == 2); // assumed by the assembly

    allzeros = containsOnlyZeros<SAMPLE_TYPE>(itsTransposedInputData->samples[station][first].origin(), nrSamples);
#else
    for (unsigned t = it->begin; allzeros && t < it->end; t++) {
      for (unsigned p = 0; p < NR_POLARIZATIONS; p++) {
        const SAMPLE_TYPE &sample = itsTransposedInputData->samples[station][t][p];

        if (real(sample) != 0.0 || imag(sample) != 0.0) {
          allzeros = false;
          break;
        }
      }
    }
#endif
  }

  if (allzeros && validSamples.count() > 0) {
    // flag everything
    SparseSet<unsigned> newflags;

    newflags.include(0, nrSamplesToCNProc);
    itsTransposedSubbandMetaData->setFlags(station, newflags);

    // Rate limit this log line, to prevent 244 warnings/station/block
    //
    // Emit (at most) one message per 10 seconds, and only one per RSP board (TODO: this doesn't work as expected with DataSlots)
    unsigned logInterval = static_cast<unsigned>(ceil(10.0 / itsCNintegrationTime));
    if (itsBlock % logInterval == 0 && *itsCurrentSubband % itsNrSlotsInFrame == 0)
      LOG_ERROR_STR(itsLogPrefix << "Station " << itsStationNames[station] << " subband " << *itsCurrentSubband << " consists of only zeros.");
  }

  timer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::dedisperseAfterBeamForming(unsigned beam, double dm)
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start dedispersion of coherent data at t = " << blockAge());
#endif

  static NSTimer timer("dedispersion (coherent) timer", true, true);

  computeTimer.start();
  timer.start();
  itsDedispersionAfterBeamForming->dedisperse(itsBeamFormedData.get(), *itsCurrentSubband, beam, dm);
  timer.stop();
  computeTimer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::preCorrelationFlagging()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start pre correlation flagger at t = " << blockAge());
#endif // HAVE_MPI

  static NSTimer timer("pre correlation flagger", true, true);

  timer.start();
  computeTimer.start();
  itsPreCorrelationFlagger->flag(itsFilteredData, *itsCurrentSubband);
  computeTimer.stop();
  timer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::preCorrelationNoChannelsFlagging()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start pre correlation no channels flagger at t = " << blockAge());
#endif // HAVE_MPI

  static NSTimer timer("pre correlation no channels flagger", true, true);

  timer.start();
  computeTimer.start();
  itsPreCorrelationNoChannelsFlagger->flag(itsFilteredData, itsBlock, *itsCurrentSubband);
  computeTimer.stop();
  timer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::mergeStations()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start merging stations at t = " << blockAge());
#endif // HAVE_MPI

  static NSTimer timer("superstation forming timer", true, true);

  timer.start();
  computeTimer.start();
  itsBeamFormer->mergeStations(itsFilteredData);
  computeTimer.stop();
  timer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::formBeams(unsigned sap, unsigned firstBeam, unsigned nrBeams)
{
  static NSTimer timer("beam forming timer", true, true);

  timer.start();
  computeTimer.start();
  itsBeamFormer->formBeams(itsTransposedSubbandMetaData, itsFilteredData, itsBeamFormedData, *itsCurrentSubband, sap, firstBeam, nrBeams);
  computeTimer.stop();
  timer.stop();

  // make sure the timer averages for forming each beam, not for forming nrBeams, a value which can be different
  // for each call to formBeams
  for (unsigned i = 1; i < nrBeams; i ++) {
    timer.start();
    timer.stop();
  }
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::correlate()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start correlating at t = " << blockAge());
#endif // HAVE_MPI

  computeTimer.start();
  itsCorrelator->computeFlags(itsFilteredData, itsCorrelatedData);
  itsCorrelator->correlate(itsFilteredData, itsCorrelatedData);
  computeTimer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::postCorrelationFlagging()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start post correlation flagger at t = " << blockAge());
#endif // HAVE_MPI

  static NSTimer timer("post correlation flagger", true, true);

  timer.start();
  computeTimer.start();
  itsPostCorrelationFlagger->flag(itsCorrelatedData, *itsCurrentSubband);

  if(itsParset.onlinePostCorrelationFlaggingDetectBrokenStations()) {
    itsPostCorrelationFlagger->detectBrokenStations();
  }

  computeTimer.stop();
  timer.stop();
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::sendOutput(StreamableData *outputData, Stream *stream)
{
#if defined HAVE_MPI
  if (LOG_CONDITION) {
    LOG_DEBUG_STR(itsLogPrefix << "Start writing output at t = " << blockAge());
  }
  //LOG_INFO_STR(itsLogPrefix << "Output " << outputNr << " has been processed " << blockAge() << " seconds after being observed.");
#endif // HAVE_MPI

  static NSTimer writeTimer("send timer", true, true);
  writeTimer.start();
  outputData->write(stream, false);
  writeTimer.stop();

  if (LOG_CONDITION) {
    LOG_DEBUG_STR(itsLogPrefix << "Done writing output at t = " << blockAge());
  }
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::finishSendingInput()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start waiting to finish sending input for transpose at t = " << blockAge());

  static NSTimer waitAsyncSendTimer("wait for all async sends", true, true);
  waitAsyncSendTimer.start();
  itsAsyncTransposeInput->waitForAllSends();
  waitAsyncSendTimer.stop();
#endif
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::finishSendingBeams()
{
#if defined HAVE_MPI
  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start waiting to finish sending beams for transpose at t = " << blockAge());

  static NSTimer waitAsyncSendTimer("wait for all async beam sends", true, true);
  waitAsyncSendTimer.start();
  itsAsyncTransposeBeams->waitForAllSends();
  waitAsyncSendTimer.stop();
#endif

  // free all pretranspose data that we just send, to make room for a different configuration
  // (because the configuration depends on itsCurrentSubband)
  for( unsigned i = 0; i < itsPreTransposeBeamFormedData.size(); i++ )
    itsPreTransposeBeamFormedData[i] = 0;
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::receiveBeam(unsigned stream)
{
#if defined HAVE_MPI
  const StreamInfo &info = itsTranspose2Logic.streamInfo[stream];
  unsigned nrSubbands = info.subbands.size();

  static NSTimer asyncFirstReceiveTimer("wait for first async beam receive", true, true);
  static NSTimer asyncNonfirstReceiveTimer("wait for subsequent async beam receive", true, true);

  if (LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Starting to receive and process subbands at t = " << blockAge());

  /* Overlap transpose and computations? */
  /* this makes timings better as this time we're waiting for data to come in
     and in a random order, so caches won't help much. In fact, we probably do
     want to process what's just been received because of those caches. */

  for (unsigned i = 0; i < nrSubbands; i++) {
    NSTimer &timer = (i == 0 ? asyncFirstReceiveTimer : asyncNonfirstReceiveTimer);
    unsigned subband;

    timer.start();
    subband = itsAsyncTransposeBeams->waitForAnyReceive();
    timer.stop();

#if 0
  /* Don't overlap transpose and computations */

    (void)subband;
  }

  for (unsigned subband = 0; subband < nrSubbands; subband++) {
#endif

    if (LOG_CONDITION && (i == 0 || i == 1 || i == nrSubbands - 2 || i == nrSubbands - 1))
      LOG_DEBUG_STR(itsLogPrefix << "Starting to post process subband " << i << " / " << nrSubbands << " at t = " << blockAge());

    if (itsFinalBeamFormedData != 0) {
      itsBeamFormer->postTransposeBeam(itsTransposedBeamFormedData, itsFinalBeamFormedData, subband, info.nrChannels, info.nrSamples);
    }  

    if (itsTrigger != 0)
      itsTrigger->compute(itsTriggerData);
  }
#else  
  (void)stream;
#endif
}


template <typename SAMPLE_TYPE> void CN_Processing<SAMPLE_TYPE>::process(unsigned block)
{
  totalProcessingTimer.start();
  NSTimer totalTimer("total processing", LOG_CONDITION, true);

  totalTimer.start();

  itsBlock = block;

  // PHASE ONE: Receive input data, and send it to the nodes participating in phase two.

#if !defined CLUSTER_SCHEDULING
  if (itsHasPhaseOne || itsHasPhaseTwo)
    transposeInput();
#endif

  // PHASE TWO: Perform (and possibly output) calculations per subband, and possibly transpose data for phase three.

  if (itsHasPhaseTwo && *itsCurrentSubband < itsNrSubbands) {
    if (LOG_CONDITION)
      LOG_DEBUG_STR(itsLogPrefix << "Phase 2: Processing subband " << *itsCurrentSubband << " block " << itsBlock << " at t = " << blockAge());

#if defined CLUSTER_SCHEDULING
    receiveInput();
#endif

    if (itsPPF != 0)
      filter();

    if (itsPreCorrelationNoChannelsFlagger != NULL)
      preCorrelationNoChannelsFlagging();

    if (itsPreCorrelationFlagger != NULL)
      preCorrelationFlagging();

#if SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG
    for(unsigned station=0; station < itsNrStations; station++) {
      fwrite(&itsBlock, sizeof(unsigned), 1, outputFile);
      fwrite(&station, sizeof(unsigned), 1, outputFile);
      int tmp = *itsCurrentSubband;
      fwrite(&tmp, sizeof(float), 1, outputFile);
      for(unsigned c=0; c<itsNrChannels; c++) {
	float sum = 0.0f;
	for(unsigned pol=0; pol < NR_POLARIZATIONS; pol++) {
	  for(unsigned t=0; t<itsNrSamplesPerIntegration; t++) {
	    fcomplex sample = itsFilteredData->samples[c][station][t][pol];
	    float power = real(sample) * real(sample) + imag(sample) * imag(sample);
	    sum += power;
	  }
	  fwrite(&sum, sizeof(float), 1, outputFile);
	}
      }
    }
    fflush(outputFile);
#endif // SAVE_REAL_TIME_FLAGGER_FILTERED_DATA_DEBUG

    mergeStations(); // create superstations
#if !defined HAVE_BGP
  }

  // transpose has to finish before we start the next transpose
  // Unlike BG/P MPI, OpenMPI performs poorly when we postpone this until
  // after correlation.

  if (itsHasPhaseOne)
    finishSendingInput();

  if (itsHasPhaseTwo && *itsCurrentSubband < itsNrSubbands) {
#endif

    if (itsCorrelator != 0)
      correlate();

    if (itsPostCorrelationFlagger != NULL)
      postCorrelationFlagging();

    if (itsCorrelatedDataStream != 0)
      sendOutput(itsCorrelatedData, itsCorrelatedDataStream);
  } 

#if defined HAVE_BGP
  if (itsHasPhaseOne) // transpose has to finish before we start the next transpose
    finishSendingInput();
#endif

  // PHASE THREE: Perform (and possibly output) calculations per beam.

  // !itsPhasThreeDisjuct: it is possible for a core not to process a subband (because *itsCurrentSubband < itsNrSubbands)
  // but has to process a beam. For instance itsNrSubbandsPerPset > nrPhase3StreamsPerPset can create such a situation: psets
  // are first filled up to itsNrSubbandsPerPset, leaving the last pset(s) idle, even though they might have to process
  // a beam.

  if ((itsHasPhaseThree && itsPhaseThreeDisjunct) || (itsHasPhaseTwo && itsPhaseThreeExists)) {
    int streamToProcess = transposeBeams(itsBlock);
    bool doPhaseThree = streamToProcess >= 0;

    if (doPhaseThree) {
      receiveBeam(streamToProcess);

      if (itsFinalBeamFormedDataStream != 0)
	sendOutput(itsFinalBeamFormedData, itsFinalBeamFormedDataStream);

      if (itsTriggerDataStream != 0)
	sendOutput(itsTriggerData, itsTriggerDataStream);
    }

    if (itsHasPhaseTwo && *itsCurrentSubband < itsNrSubbands)
      finishSendingBeams();
  }

#if defined HAVE_MPI
  if ((itsHasPhaseOne || itsHasPhaseTwo || itsHasPhaseThree) && LOG_CONDITION)
    LOG_DEBUG_STR(itsLogPrefix << "Start idling at t = " << blockAge());
#endif // HAVE_MPI

#if 0
  static unsigned count = 0;

  if (itsLocationInfo.rank() == 5 && ++ count == 9)
    for (double time = MPI_Wtime() + 4.0; MPI_Wtime() < time;)
      ;
#endif

  if (itsHasPhaseTwo)
    itsCurrentSubband->next();

  totalTimer.stop();
  totalProcessingTimer.stop();
}


template class CN_Processing<i4complex>;
template class CN_Processing<i8complex>;
template class CN_Processing<i16complex>;

} // namespace RTCP
} // namespace LOFAR
