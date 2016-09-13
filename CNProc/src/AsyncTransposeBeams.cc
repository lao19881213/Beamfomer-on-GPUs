//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <AsyncTransposeBeams.h>

#include <Interface/CN_Mapping.h>
#include <Interface/PrintVector.h>
#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>

#include <cassert>

//#define DEBUG

namespace LOFAR {
namespace RTCP {

#if defined HAVE_MPI


union Tag {
  struct {
    unsigned sign       : 1; /* must be 0 */
    unsigned sourceRank :11; /* 0..4095, or one BG/P rack */
    unsigned subband    :10;
    unsigned beam       : 9;
  } info;

  uint32 nr;

  Tag(): nr(0) {}
};

AsyncTransposeBeams::AsyncTransposeBeams(
  bool isTransposeInput, bool isTransposeOutput, unsigned nrSubbands,
  const LocationInfo &locationInfo,
  const std::vector<unsigned> &inputPsets, const std::vector<unsigned> &inputCores, const std::vector<unsigned> &outputPsets, const std::vector<unsigned> &outputCores )
:
  itsIsTransposeInput(isTransposeInput),
  itsIsTransposeOutput(isTransposeOutput),
  itsAsyncComm(),
  itsInputPsets(inputPsets),
  itsInputCores(inputCores),
  itsOutputPsets(outputPsets),
  itsOutputCores(outputCores),
  itsLocationInfo(locationInfo),
  itsCommHandles(itsNrCommunications,nrSubbands),
  itsLocalSubbands(nrSubbands)
{
  ASSERT(itsNrCommunications == 1); // no bits left to encode communication channel, so we can support only one
}

  template <typename T,unsigned DIM, unsigned FLAGS_DIM> void AsyncTransposeBeams::postReceive(SampleData<T,DIM,FLAGS_DIM> *transposedData, unsigned localSubband, unsigned globalSubband, unsigned beam, unsigned psetIndex, unsigned coreIndex)
{
  unsigned pset = itsInputPsets[psetIndex];
  unsigned core = itsInputCores[coreIndex];

  unsigned rank = itsLocationInfo.remapOnTree(pset, core); // TODO cache this? maybe in locationInfo itself?

  // define what to read
  struct {
    void   *ptr;
    size_t size;
  } toRead[itsNrCommunications] = {
    { transposedData->samples[localSubband].origin(), transposedData->samples[localSubband].num_elements() * sizeof(T) }
  };

  itsLocalSubbands[globalSubband] = localSubband;

  // read it
  for (unsigned h = 0; h < itsNrCommunications; h ++) {
    Tag t;

    t.info.sign       = 0;
    t.info.sourceRank = rank;
    //t.info.comm       = h;
    t.info.beam       = beam;
    t.info.subband    = globalSubband;

#ifdef DEBUG
    LOG_DEBUG_STR( "Posting to receive beam " << beam << " subband " << globalSubband << " (local: subband " << localSubband << ") from pset " << pset << " core " << core << " = rank " << rank << ", tag " << t.nr );
#endif
    itsCommHandles[h][globalSubband] = itsAsyncComm.asyncRead(toRead[h].ptr, toRead[h].size, rank, t.nr);
  }
}

// returns station number (= pset index)
unsigned AsyncTransposeBeams::waitForAnyReceive()
{
  while (true) {
    void     *buf;
    unsigned size, source;
    int      tag;

    Tag t;

    // This read could return any type of message (out of itsCommHandles)
    itsAsyncComm.waitForAnyRead(buf, size, source, tag);

    t.nr = tag;

    unsigned subband = t.info.subband;
#ifdef DEBUG
    unsigned rank = t.info.sourceRank;
    LOG_DEBUG_STR( "Received subband " << subband << " from pset ??, rank " << rank << ", tag " << tag );
#endif
    // mark the right communication handle as received
    unsigned comm = 0; // = t.info.comm;
    itsCommHandles[comm][subband] = -1;

    // check whether we have received all communications for this psetIndex.
    // This is the case when commHandles are -1.
    bool haveAll = true;

    for (unsigned h = 0; h < itsNrCommunications; h ++) {
      if (itsCommHandles[h][subband] != -1) {
        haveAll = false;
        break;
      }
    }

    if (haveAll)
      return itsLocalSubbands[subband];
  }
}


  template <typename T, unsigned DIM, unsigned FLAGS_DIM> void AsyncTransposeBeams::asyncSend(unsigned outputPsetIndex, unsigned coreIndex, unsigned subband, unsigned stokes, unsigned globalBeam, const SampleData<T,DIM,FLAGS_DIM> *inputData)
{
  unsigned pset = itsOutputPsets[outputPsetIndex];
  unsigned core = itsOutputCores[coreIndex];
  unsigned rank = itsLocationInfo.remapOnTree(pset, core);

  // define what to write
  struct {
    const void   *ptr;
    const size_t size;
  } toWrite[itsNrCommunications] = {
    { inputData->samples[stokes].origin(), inputData->samples[stokes].num_elements() * sizeof(T) }
  };

  // write it
  for (unsigned h = 0; h < itsNrCommunications; h ++) {
    Tag t;

    t.info.sign       = 0;
    t.info.sourceRank = itsLocationInfo.rank();
    //t.info.comm       = h;
    t.info.subband    = subband;
    t.info.beam       = globalBeam;

#ifdef DEBUG
    LOG_DEBUG_STR( "Sending beam " << globalBeam << " (local: stokes " << stokes << ") subband " << subband << " to pset " << pset << " core " << core << " = rank " << rank << ", tag " << t.nr );
#endif
    itsAsyncComm.asyncWrite(toWrite[h].ptr, toWrite[h].size, rank, t.nr);
  }
}

template void AsyncTransposeBeams::postReceive(SampleData<float,3,1> *, unsigned, unsigned, unsigned, unsigned, unsigned);
template void AsyncTransposeBeams::postReceive(SampleData<float,3,2> *, unsigned, unsigned, unsigned, unsigned, unsigned);
template void AsyncTransposeBeams::asyncSend(unsigned, unsigned, unsigned, unsigned, unsigned, const SampleData<float,3,1> *);

void AsyncTransposeBeams::waitForAllSends()
{
  // this includes the metadata writes...
  itsAsyncComm.waitForAllWrites();
}


#endif // HAVE_MPI

} // namespace RTCP
} // namespace LOFAR
