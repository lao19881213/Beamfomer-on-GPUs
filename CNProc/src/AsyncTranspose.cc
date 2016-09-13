//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <AsyncTranspose.h>

#include <Interface/CN_Mapping.h>
#include <Interface/PrintVector.h>
#include <Common/LofarLogger.h>

#include <cassert>


namespace LOFAR {
namespace RTCP {

#if defined HAVE_MPI

#define MAX_RANK 100000 // used for message identification: id = type*MAX_RANK + rank

template <typename SAMPLE_TYPE> AsyncTranspose<SAMPLE_TYPE>::AsyncTranspose(
  bool isTransposeInput, bool isTransposeOutput, 
  unsigned groupNumber, const LocationInfo &locationInfo, 
  const std::vector<unsigned> &inputPsets, const std::vector<unsigned> &outputPsets )
:
  itsIsTransposeInput(isTransposeInput),
  itsIsTransposeOutput(isTransposeOutput),
  itsAsyncComm(),
  itsInputPsets(inputPsets),
  itsOutputPsets(outputPsets),
  itsLocationInfo(locationInfo),
  itsCommHandles(itsNrCommunications, inputPsets.size()),
  itsGroupNumber(groupNumber)
{
  for (unsigned i = 0; i < inputPsets.size(); i ++) {
    unsigned rank = locationInfo.remapOnTree(inputPsets[i], itsGroupNumber);

    itsRankToPsetIndex[rank] = i;
  }
}

template <typename SAMPLE_TYPE> void AsyncTranspose<SAMPLE_TYPE>::postAllReceives(SubbandMetaData *metaData, TransposedData<SAMPLE_TYPE> *transposedData)
{
  // there must be something to receive
  ASSERT(itsInputPsets.size() > 0);

  for (unsigned i = 0; i < itsInputPsets.size(); i ++) {
    unsigned pset = itsInputPsets[i];
    unsigned rank = itsLocationInfo.remapOnTree(pset, itsGroupNumber); // TODO cache this? maybe in locationInfo itself?

    // define what to read
    struct {
      void   *ptr;
      size_t size;
    } toRead[itsNrCommunications] = {
      { transposedData->samples[i].origin(), transposedData->samples[i].num_elements() * sizeof(SAMPLE_TYPE) },
      { &metaData->subbandInfo(i), metaData->itsSubbandInfoSize }
    };

    // read it
    for (unsigned h = 0; h < itsNrCommunications; h ++)
      itsCommHandles[h][i] = itsAsyncComm.asyncRead(toRead[h].ptr, toRead[h].size, rank, rank + h * MAX_RANK);
  }
}


// returns station number (= pset index)
template <typename SAMPLE_TYPE> unsigned AsyncTranspose<SAMPLE_TYPE>::waitForAnyReceive()
{
  while (true) {
    void     *buf;
    unsigned size, source;
    int      tag;

    // This read could return any type of message (out of itsCommHandles)
    itsAsyncComm.waitForAnyRead(buf, size, source, tag);

    // source is the real rank, calc pset index
    unsigned psetIndex = itsRankToPsetIndex[source];

    // mark the right communication handle as received
    for (unsigned h = 0; h < itsNrCommunications; h ++) {
      if (static_cast<unsigned>(tag) < (h + 1) * MAX_RANK) {
        itsCommHandles[h][psetIndex] = -1;
        break;
      }
    }

    // check whether we have received all communications for this psetIndex.
    // This is the case when commHandles are -1.
    bool haveAll = true;

    for (unsigned h = 0; h < itsNrCommunications; h ++) {
      if (itsCommHandles[h][psetIndex] != -1) {
        haveAll = false;
        break;
      }
    }

    if (haveAll)
      return psetIndex;
  }
}


template <typename SAMPLE_TYPE> void AsyncTranspose<SAMPLE_TYPE>::asyncSend(unsigned outputPsetIndex, 
                                                                            const SubbandMetaData *metaData,
									    const InputData<SAMPLE_TYPE> *inputData)
{
  unsigned pset = itsOutputPsets[outputPsetIndex];
  unsigned rank = itsLocationInfo.remapOnTree(pset, itsGroupNumber);
  int	   tag  = itsLocationInfo.rank();

  // define what to write
  struct {
    const void *ptr;
    size_t     size;
  } toWrite[itsNrCommunications] = {
    { inputData->samples[outputPsetIndex].origin(), inputData->samples[outputPsetIndex].num_elements() * sizeof(SAMPLE_TYPE) },
    { &metaData->subbandInfo(outputPsetIndex), metaData->itsSubbandInfoSize },
  };

  // write it
  for (unsigned h = 0; h < itsNrCommunications; h ++)
    itsAsyncComm.asyncWrite(toWrite[h].ptr, toWrite[h].size, rank, tag + h * MAX_RANK);
}


template <typename SAMPLE_TYPE> void AsyncTranspose<SAMPLE_TYPE>::waitForAllSends()
{
  // this includes the metadata writes...
  itsAsyncComm.waitForAllWrites();
}

  
template class AsyncTranspose<i4complex>;
template class AsyncTranspose<i8complex>;
template class AsyncTranspose<i16complex>;

#endif // HAVE_MPI

} // namespace RTCP
} // namespace LOFAR
