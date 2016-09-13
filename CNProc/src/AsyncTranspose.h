#ifndef LOFAR_CNPROC_ASYNC_TRANSPOSE_H
#define LOFAR_CNPROC_ASYNC_TRANSPOSE_H

#include <AsyncCommunication.h>
#include <Interface/InputData.h>
#include <LocationInfo.h>
#include <Interface/TransposedData.h>
#include <Interface/SubbandMetaData.h>

#if defined HAVE_MPI
#define MPICH_IGNORE_CXX_SEEK
#include <mpi.h>
#endif

#include <vector>

namespace LOFAR {
namespace RTCP {

#if defined HAVE_MPI

// Nodes in input psets read outputPsets.size subbands from their I/O node (one by one).
// Cores communicate with the same logical core number in another pset 
// (due to an extra mapping, this is not the physical core number).

// # sends = size outputPsets (= nrSubbands) on the input nodes.
// # recvs = size inputPsets (= nrStations) on the output nodes.
// Only the output nodes are actually calculating (filtering and correlating).

template <typename SAMPLE_TYPE> class AsyncTranspose
{
  public:

  AsyncTranspose(bool isTransposeInput, bool isTransposeOutput, 
		 unsigned groupNumber, const LocationInfo &, 
		 const std::vector<unsigned> &inputPsets, const std::vector<unsigned> &outputPsets );
  
  // Post all async receives for the transpose.
  void postAllReceives(SubbandMetaData *metaData, TransposedData<SAMPLE_TYPE> *transposedData);
  
  // Wait for a data message. Returns the station number where the message originates.
  unsigned waitForAnyReceive();
  
  // Asynchronously send a subband.
  void asyncSend(unsigned outputPsetNr, const SubbandMetaData *metaData, const InputData<SAMPLE_TYPE> *inputData);
  
  // Make sure all async sends have finished.
  void waitForAllSends();
  
 private:
  
  const bool itsIsTransposeInput, itsIsTransposeOutput;

  unsigned itsNrSubbands;
  unsigned itsNrSubbandsPerPset;
  unsigned itsNrTABs;

  // A mapping that tells us, if we receive a message from a source,
  // to which pset that source belongs.
  std::map<unsigned, unsigned> itsRankToPsetIndex; 

  AsyncCommunication itsAsyncComm;
  const std::vector<unsigned> itsInputPsets;
  const std::vector<unsigned> itsOutputPsets;
  const LocationInfo &itsLocationInfo;

  // The number of communicates (writes/reads) needed to transport one sub band.
  static const unsigned itsNrCommunications = 2;

  // The maps that contain the handles to the asynchronous reads.
  // The maps are indexed by the inputPset index.
  // The value is -1 if the read finished.
  Matrix<int> itsCommHandles; // [itsNrCommunications][itsNrInputPsets]

  // The number of the transpose group we belong to.
  // The cores with the same index in a pset together form a group.
  unsigned itsGroupNumber;
};

#endif // defined HAVE_MPI

} // namespace RTCP
} // namespace LOFAR

#endif
