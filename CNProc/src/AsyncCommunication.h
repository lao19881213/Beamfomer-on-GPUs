#ifndef LOFAR_CNPROC_ASYNC_COMMUNICATION_H
#define LOFAR_CNPROC_ASYNC_COMMUNICATION_H

#if defined HAVE_MPI
#define MPICH_IGNORE_CXX_SEEK
#include <mpi.h>
#endif

#include <map>
#include <boost/noncopyable.hpp>

namespace LOFAR {
namespace RTCP {

#if defined HAVE_MPI

class AsyncRequest {
public:
  MPI_Request mpiReq;
  void* buf;
  unsigned size;
  unsigned rank;
  int tag;
};

class AsyncCommunication: boost::noncopyable {
  public:
    AsyncCommunication(MPI_Comm communicator = MPI_COMM_WORLD);

    // returns handle to this read
    int asyncRead(void* buf, unsigned size, unsigned source, int tag);

    // returns handle to this write
    int asyncWrite(const void* buf, unsigned size, unsigned dest, int tag);

    void waitForRead(int handle);
    void waitForWrite(int handle);

    // returns the handle of the read that was done.
    int waitForAnyRead(void*& buf, unsigned& size, unsigned& source, int& tag);

    void waitForAllReads();
    void waitForAllWrites();
			      
private:

    MPI_Comm itsCommunicator;
    int itsCurrentReadHandle;
    int itsCurrentWriteHandle;
    std::map<int, AsyncRequest*> itsReadHandleMap;
    std::map<int, AsyncRequest*> itsWriteHandleMap;
};

#endif // defined HAVE_MPI

} // namespace RTCP
} // namespace LOFAR

#endif
