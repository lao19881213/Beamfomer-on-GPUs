#ifndef LOFAR_INPUTPROC_MPIUTIL_H
#define LOFAR_INPUTPROC_MPIUTIL_H

#include <mpi.h>
#include <vector>

#include <Common/Thread/Mutex.h>
#include <CoInterface/Allocator.h>
#include <CoInterface/SmartPtr.h>

namespace LOFAR {

  namespace Cobalt {

    // MPI routines lock this mutex, unless annotated with "NOT LOCKED"
    extern Mutex MPIMutex;

    // Return the MPI rank of this process
    int MPI_Rank();

    // Return the number of MPI ranks in this run
    int MPI_Size();

    // Return whether MPI is initialised
    bool MPI_Initialised();

    // An allocator using MPI_Alloc_mem/MPI_Free_mem
    class MPIAllocator : public Allocator
    {
    public:
      virtual void                *allocate(size_t size, size_t alignment = 1);
      virtual void                deallocate(void *);
    };

    extern MPIAllocator mpiAllocator;

    // Free functor for CoInterface/SmartPtr
    template <typename T = void>
    class SmartPtrMPI
    {
    public:
      static void free( T *ptr )
      {
        mpiAllocator.deallocate(ptr);
      }
    };

    // Wait for any request to finish. Returns the index of the request that
    // finished. Finished requests are set to MPI_REQUEST_NULL and ignored in
    // subsequent calls.
    int waitAny( std::vector<MPI_Request> &requests );

    // Wait for all given requests to finish. Finished requests are set to
    // MPI_REQUEST_NULL and ignored in subsequent calls.
    void waitAll( std::vector<MPI_Request> &requests );

    // Free an MPI request (do not wait for it to finish).
    //
    // NOT LOCKED
    void freeRequest(MPI_Request &request);

    // Wait for any number of given requests to finish. Returns the indices of
    // the requests that finished. Finished requests are set to
    // MPI_REQUEST_NULL and ignored in subsequent calls.
    std::vector<int> waitSome( std::vector<MPI_Request> &requests );

    /*
     * A guarded version of MPI_Issend with fewer parameters.
     *
     * Issend: Waits for receive to be posted before sending data,
     *         allowing direct transfers/preventing copying.
     *
     * NOT LOCKED
     */
    MPI_Request Guarded_MPI_Issend(const void *ptr, size_t numBytes, int destRank, int tag);

    /*
     * A guarded version of MPI_Ibsend with fewer parameters.
     *
     * Ibsend: Requires user buffers to be set up (MPI_Buffer_init)
     *
     * NOT LOCKED
     */
    MPI_Request Guarded_MPI_Ibsend(const void *ptr, size_t numBytes, int destRank, int tag);
    /*
     * A guarded version of MPI_Irsend with fewer parameters.
     *
     * Irsend: Requires matching receive to have been posted.
     *
     * NOT LOCKED
     */
    MPI_Request Guarded_MPI_Irsend(const void *ptr, size_t numBytes, int destRank, int tag);

    /*
     * A guarded version of MPI_Isend with fewer parameters.
     *
     * Isend: Allows MPI to chose the best send method (buffer or sync)
     *
     * NOT LOCKED
     */
    MPI_Request Guarded_MPI_Isend(const void *ptr, size_t numBytes, int destRank, int tag);

    /*
     * A guarded version of MPI_Irecv with fewer parameters.
     *
     * NOT LOCKED
     */
    MPI_Request Guarded_MPI_Irecv(void *ptr, size_t numBytes, int srcRank, int tag);
  }

}

#endif

