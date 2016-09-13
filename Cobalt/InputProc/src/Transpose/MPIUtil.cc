#include <lofar_config.h>
#include "MPIUtil.h"

/*
 * New plan:
 *
 *
 * 1) One thread / subband.
 *    - will result in thousands of threads!
 *    - better: one thread / dest node?
 *
 * 2) One polling thread / process.
 *
 * Design for 1):
 *
 * Design for 2):
 *
 * 1. Polling thread does:

     // Wait request as provided by clients
     struct request_cl {
       // MPI handle
       int handle;

       // Whether the request has been satisfied
       bool done;

       // Whether the request has been reported
       // to the client (waitAny)
       bool reported;

       request_cl(int handle): handle(handle), done(false), reported(false) {}
     };

     // Wait request as maintained by pollThread
     struct request {
       // MPI handle
       int handle;

       // Pointer to request_cl.done,
       // indicating whether the request
       // has been satisfied.
       bool *done;

       // Signal to trigger if the request
       // satisfies.
       Semaphore *doneSignal;
     };

     vector<request> requests;
     Mutex requestMutex;
     Condition newRequest;

     void Testsome( vector<int> &doneset, vector<int> &handles ) {
       const size_t incount = requests.size();

       handles.resize(incount);
       doneset.resize(incount);

       for (size_t i = 0; i < incount; ++i)
         handles[i] = requests[i].handle;

       int outcount;
       MPI_Testsome(incount, &handles[0], &outcount, &doneset[0], MPI_STATUSES_IGNORE);
       doneset.resize(outcount);

       return doneset;
     }

     void pollThread() {
       ScopedLock sl(requestMutex);

       // cache doneset/handles, for efficiency.
       // Testsome() handles these.
       vector<int> doneset;
       vector<int> handles;

       while(!done) {
         if (requests.empty()) {
           // wait for request, with lock released
           newRequest.wait(requestMutex);
         } else {
           // poll existing requests
           Testsome(doneset, handles);

           // set of wait* functions we've signaled
           set<Semaphore*> signalSet;

           // mark finished requests,
           // and trigger each of their
           // signals at most once
           for (size_t i = 0; i < doneset.size(); ++i) {
             struct request &r = requests[doneset[i]];

             // mark request_cl as done
             r.handle = MPI_REQUEST_NULL; // don't give this one to MPI_Testsome again
             *(r.done) = true;

             if (signalSet.add(r.doneSignal).second) {
               // signal was not yet in set -- trigger
               r.doneSignal->signal();
             }
           }

           // if there are still pending requests, release
           // the lock and just wait with a timeout
           if (!requests.empty()) {
             newRequest.wait(requestMutex, now() + 0.001);
           }
         }
       }
     }

     // Register/unregister requests in bulk to reduce overhead

     void registerRequests( const std::vector<request_cl> &newRequests, Semaphore &doneSignal ) {
       ScopedLock sl(requestMutex);

       // register all new requests
       requests.reserve(requests.size() + newRequests.size());
       for (size_t i = 0; i < newRequests.size(); ++i) {
         struct request_cl &from = newRequests[i];
         struct request to = { from.handle, &from.done, &doneSignal };
         requests.push_back(to);
       }

       // trigger our thread
       newRequest.signal();
     }

     void unregisterRequests( Semaphore &doneSignal ) {
       ScopedLock sl(requestMutex);

       // unregister all requests
       vector<struct request> writePtr = requests.begin();
       for (size_t i = 0; i < requests.size(); ++i) {
         if (requests[i].doneSignal != &doneSignal) {
           // keep
           *(writePtr++) = requests[i];
         } else {
           // ours -- throw away
           ASSERTSTR(requests[i].done, "Unregistering an uncompleted request");
         }
       }

       // cut off at new size
       requests.resize(writePtr - requests.begin());
     }

     size_t waitAny( const std::vector<request_cl> &waitRequests, Semaphore &doneSignal ) {
       // assumes all waitRequests have been registered

       for(;;) {
         {
           ScopedLock sl(requestMutex);

           // check if any request has finished already
           for (size_t i = 0; i < waitRequests.size(); ++i) {
             struct request_cl &r = waitRequests[i];

             if (r.done && !r.reported) {
               r.reported = true;

               return i;
             }
           }
         }

         // wait for a request to finish
         doneSignal.wait();
       }
     }

     void waitAll( const std::vector<request_cl> &waitRequests, Semaphore &doneSignal ) {
       // assumes all waitRequests have been registered

       for(;;) {
         {
           ScopedLock sl(requestMutex);

           bool allDone = true;

           // check if any request is still pending
           for (size_t i = 0; i < waitRequests.size(); ++i) {
             struct request_cl &r = waitRequests[i];

             if (!r.done) {
               allDone = false;
               break;
             }
           }

           if (allDone)
             return;
         }

         // wait for a request to finish
         doneSignal.wait();
       }
     }
 */

#include <iomanip>

#include <Common/LofarLogger.h>

//#define DEBUG_MPI

#ifdef DEBUG_MPI
#define DEBUG(str)  LOG_DEBUG_STR(str)
#else
#define DEBUG(str)
#endif

#include <ctime>
#include <Common/Thread/Mutex.h>
#include <CoInterface/SmartPtr.h>

using namespace std;

namespace LOFAR {

  namespace Cobalt {

    Mutex MPIMutex;

    int MPI_Rank() {
      ScopedLock sl(MPIMutex);
      
      int rank;

      ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      return rank;
    }

    int MPI_Size() {
      ScopedLock sl(MPIMutex);
      
      int size;

      ::MPI_Comm_size(MPI_COMM_WORLD, &size);

      return size;
    }

    bool MPI_Initialised() {
      int flag;

      ::MPI_Initialized(&flag);

      return flag != 0;
    }


    void *MPIAllocator::allocate(size_t size, size_t alignment)
    {
      ScopedLock sl(MPIMutex);

      ASSERT(alignment == 1); // Don't support anything else yet, although MPI likely aligns for us
      ASSERT(MPI_Initialised());

      void *ptr;

      int error = ::MPI_Alloc_mem(size, MPI_INFO_NULL, &ptr);
      ASSERT(error == MPI_SUCCESS);

      return ptr;
    }


    void MPIAllocator::deallocate(void *ptr)
    {
      ScopedLock sl(MPIMutex);

      int error = ::MPI_Free_mem(ptr);
      ASSERT(error == MPI_SUCCESS);
    }

    MPIAllocator mpiAllocator;

    void freeRequest(MPI_Request &request) {
      ::MPI_Request_free(&request);
      ASSERT(request == MPI_REQUEST_NULL);
    }

    /*
     * Returns (and caches) whether the MPI library is thread safe.
     */
    static bool MPI_threadSafe() {
      return false;
      /*
       * The level of threading support as reported by the MPI library.
       *
       * For multi-threading support, we need at least MPI_THREAD_MULTIPLE.
       *
       * Note that OpenMPI will claim to support the safe-but-useless
       * MPI_THREAD_SINGLE, while it supports MPI_THREAD_SERIALIZED in practice.
       */
      static int provided_mpi_thread_support;
      static const int minimal_thread_level = MPI_THREAD_MULTIPLE;

      static bool initialised = false;

      if (!initialised) {
        ScopedLock sl(MPIMutex);

        int error = MPI_Query_thread(&provided_mpi_thread_support);
        ASSERT(error == MPI_SUCCESS);

        initialised = true;

        LOG_INFO_STR("MPI is thread-safe: " << (provided_mpi_thread_support >= minimal_thread_level ? "yes" : "no"));
      }

      return provided_mpi_thread_support >= minimal_thread_level;
    }

    std::vector<int> waitSome( std::vector<MPI_Request> &requests )
    {
      DEBUG("entry");

      if (requests.empty())
        return vector<int>(0);

      std::vector<int> completed(requests.size());
      int nr_completed = 0;

      if (MPI_threadSafe()) {
        int error = MPI_Waitsome(requests.size(), &requests[0], &nr_completed, &completed[0], MPI_STATUSES_IGNORE);
        ASSERT(error == MPI_SUCCESS);
      } else {
        do {
          {
            ScopedLock sl(MPIMutex);

            int error = MPI_Testsome(requests.size(), &requests[0], &nr_completed, &completed[0], MPI_STATUSES_IGNORE);
            ASSERT(error == MPI_SUCCESS);
          }

          // sleep (with lock released)
          if (nr_completed == 0) {
            const struct timespec req = { 0, 10000000 }; // 10 ms
            nanosleep(&req, NULL);
          }
        } while(nr_completed == 0);
      }

      ASSERT(nr_completed != MPI_UNDEFINED);
      ASSERT(nr_completed >= 0);
      ASSERT((size_t)nr_completed < requests.size());

      // cut off array
      completed.resize(nr_completed);

      return completed;
    }

    int waitAny( std::vector<MPI_Request> &requests )
    {
      DEBUG("entry");

      int idx;

      if (MPI_threadSafe()) {
        int error = MPI_Waitany(requests.size(), &requests[0], &idx, MPI_STATUS_IGNORE);
        ASSERT(error == MPI_SUCCESS);
      } else {
        int flag;

        do {
          {
            ScopedLock sl(MPIMutex);

            int error = MPI_Testany(requests.size(), &requests[0], &idx, &flag, MPI_STATUS_IGNORE);
            ASSERT(error == MPI_SUCCESS);
          }

          // sleep (with lock released)
          if (!flag) {
            const struct timespec req = { 0, 10000000 }; // 10 ms
            nanosleep(&req, NULL);
          }
        } while(!flag);
      }

      ASSERT(idx != MPI_UNDEFINED);

      // NOTE: MPI_Waitany/MPI_Testany will overwrite completed
      // entries with MPI_REQUEST_NULL, unless the request
      // was persistent (MPI_Send_init + MPI_Start).
      ASSERT(requests[idx] == MPI_REQUEST_NULL);

      DEBUG("index " << idx);

      return idx;
    }


    void waitAll( std::vector<MPI_Request> &requests )
    {
      DEBUG("entry: " << requests.size() << " requests");

      if (requests.size() > 0) {
        if (MPI_threadSafe()) {
          int error = MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
          ASSERT(error == MPI_SUCCESS);
        } else {
          int flag;

          do {
            {
              ScopedLock sl(MPIMutex);

              int error = MPI_Testall(requests.size(), &requests[0], &flag, MPI_STATUSES_IGNORE);
              ASSERT(error == MPI_SUCCESS);
            }

            // sleep (with lock released)
            if (!flag) {
              const struct timespec req = { 0, 10000000 }; // 10 ms
              nanosleep(&req, NULL);
            }
          } while(!flag);
        }
      }

      // NOTE: MPI_Waitall/MPI_Testall will overwrite completed
      // entries with MPI_REQUEST_NULL.

      DEBUG("exit");
    }

    namespace {
      typedef int (*MPI_SEND)(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*);

      // Generic send method
      MPI_Request Guarded_MPI_Send(MPI_SEND sendMethod, const void *ptr, size_t numBytes, int destRank, int tag) {
        DEBUG("SEND: size " << numBytes << " tag " << hex << tag << dec << " to " << destRank);

        ASSERT(numBytes > 0);
        ASSERT(tag >= 0); // Silly MPI requirement

        //SmartPtr<ScopedLock> sl = MPI_threadSafe() ? 0 : new ScopedLock(MPIMutex);

        MPI_Request request;

        int error;

        error = sendMethod(const_cast<void*>(ptr), numBytes, MPI_BYTE, destRank, tag, MPI_COMM_WORLD, &request);
        ASSERT(error == MPI_SUCCESS);

        return request;
      }
    }


    MPI_Request Guarded_MPI_Issend(const void *ptr, size_t numBytes, int destRank, int tag) {
      return Guarded_MPI_Send(::MPI_Issend, ptr, numBytes, destRank, tag);
    }

    MPI_Request Guarded_MPI_Irsend(const void *ptr, size_t numBytes, int destRank, int tag) {
      return Guarded_MPI_Send(::MPI_Irsend, ptr, numBytes, destRank, tag);
    }

    MPI_Request Guarded_MPI_Ibsend(const void *ptr, size_t numBytes, int destRank, int tag) {
      return Guarded_MPI_Send(::MPI_Ibsend, ptr, numBytes, destRank, tag);
    }

    MPI_Request Guarded_MPI_Isend(const void *ptr, size_t numBytes, int destRank, int tag) {
      return Guarded_MPI_Send(::MPI_Isend, ptr, numBytes, destRank, tag);
    }

    MPI_Request Guarded_MPI_Irecv(void *ptr, size_t numBytes, int srcRank, int tag) {
      DEBUG("RECV: size " << numBytes << " tag " << hex << tag);

      ASSERT(tag >= 0); // Silly MPI requirement

      //SmartPtr<ScopedLock> sl = MPI_threadSafe() ? 0 : new ScopedLock(MPIMutex);

      MPI_Request request;

      int error;

      error = ::MPI_Irecv(ptr, numBytes, MPI_BYTE, srcRank, tag, MPI_COMM_WORLD, &request);
      ASSERT(error == MPI_SUCCESS);

      return request;
    }

  }
}

