//# Always <lofar_config.h> first!
#include <lofar_config.h>

#include <AsyncCommunication.h>

#include <Common/Timer.h>
#include <Interface/Exceptions.h>

#include <cassert>
#include <map>
#include <iostream>

#include <boost/format.hpp>

#define USE_TIMING 0

namespace LOFAR {
namespace RTCP {

#if defined HAVE_MPI

// convert an MPI return code into an error string
static string errorstr(int res)
{
  int eclass, len;
  char estring[MPI_MAX_ERROR_STRING];
  MPI_Error_class(res, &eclass);
  MPI_Error_string(res, estring, &len);

  // convert newlines to spaces to keep the message on a single line
  for (char *c = estring; *c; c ++)
    if (*c == '\n')
      *c = ' ';

  return str(boost::format("error %d: %s") % eclass % estring);
}

AsyncCommunication::AsyncCommunication(MPI_Comm comm)
:
  itsCommunicator(comm),
  itsCurrentReadHandle(0),
  itsCurrentWriteHandle(0)
{
}

// returns handle to this read
int AsyncCommunication::asyncRead(void* buf, unsigned size, unsigned source, int tag)
{
  AsyncRequest *req = new AsyncRequest();

  int res = MPI_Irecv(buf, size, MPI_BYTE, source, tag, itsCommunicator, &req->mpiReq);

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Irecv() failed: " << errorstr(res));

  req->buf = buf;
  req->size = size;
  req->rank = source;
  req->tag = tag;

  int handle = itsCurrentReadHandle++;
  itsReadHandleMap[handle] = req;

  return handle;
}

// returns handle to this write
int AsyncCommunication::asyncWrite(const void* buf, unsigned size, unsigned dest, int tag)
{
  AsyncRequest *req = new AsyncRequest();

  int res = MPI_Isend(const_cast<void*>(buf), size, MPI_BYTE, dest, tag, itsCommunicator, &req->mpiReq);

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Isend() failed: " << errorstr(res));

  req->buf = (void*)buf;
  req->size = size;
  req->rank = dest;
  req->tag = tag;

  int handle = itsCurrentWriteHandle++;
  itsWriteHandleMap[handle] = req;

  return handle;
}


void AsyncCommunication::waitForRead(int handle)
{
  AsyncRequest *req = itsReadHandleMap[handle];
  MPI_Status status;

  int res = MPI_Wait(&req->mpiReq, &status);

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Wait() failed: " << errorstr(res));

  // done, now remove from map, and free req
  itsReadHandleMap.erase(handle);
  delete req;
}


void AsyncCommunication::waitForWrite(int handle)
{
  AsyncRequest *req = itsWriteHandleMap[handle];
  MPI_Status status;

  int res = MPI_Wait(&req->mpiReq, &status);

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Wait() failed: " << errorstr(res));

  // done, now remove from map, and free req
  itsWriteHandleMap.erase(handle);
  delete req;
}


// returns the handle of the read that was done.
int AsyncCommunication::waitForAnyRead(void*& buf, unsigned& size, unsigned& source, int& tag)
{
  MPI_Status status;
  int count = itsReadHandleMap.size();
  MPI_Request reqs[count];
  int mapping[count];

  ASSERT( count > 0 );
  
  int i = 0;

  for (std::map<int, AsyncRequest*>::const_iterator it = itsReadHandleMap.begin(); it != itsReadHandleMap.end(); it ++) {
    int handle = it->first;
    AsyncRequest* r = it->second;
    reqs[i] = r->mpiReq;
    mapping[i] = handle;
    i ++;
  }

  NSTimer waitAnyTimer("MPI_Waitany", USE_TIMING, true);
  waitAnyTimer.start();
  int index = -1;
  int res = MPI_Waitany(count, reqs, &index, &status);
  waitAnyTimer.stop();

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Waitany() failed: " << errorstr(res));

  if (index == MPI_UNDEFINED)
    THROW(CNProcException,"MPI_Waitany() failed: no (pending) receives");

  int handle = mapping[index];
  AsyncRequest* req = itsReadHandleMap[handle];

  buf = req->buf;
  size = req->size;
  source = req->rank;
  tag = req->tag;

  itsReadHandleMap.erase(handle);
  delete req;
  return handle;
}


void AsyncCommunication::waitForAllReads()
{
  int count = itsReadHandleMap.size();
  MPI_Request reqs[count];
  MPI_Status status[count];

  if (count == 0)
    return; // nothing to wait for

  int i = 0;

  for (std::map<int, AsyncRequest*>::const_iterator it = itsReadHandleMap.begin(); it != itsReadHandleMap.end(); it ++) {
    AsyncRequest* r = it->second;
    reqs[i] = r->mpiReq;
    i ++;
  }

  int res = MPI_Waitall(count, reqs, status);

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Waitall() failed: " << errorstr(res));

  for (std::map<int, AsyncRequest*>::const_iterator it = itsReadHandleMap.begin(); it != itsReadHandleMap.end(); it ++) {
    AsyncRequest *r = it->second;
    delete r;
  }

  itsReadHandleMap.clear();
  itsCurrentReadHandle = 0;
}


void AsyncCommunication::waitForAllWrites()
{
  int count = itsWriteHandleMap.size();
  MPI_Request reqs[count];
  MPI_Status status[count];

  if (count == 0)
    return; // nothing to wait for

  int i = 0;

  for (std::map<int, AsyncRequest*>::const_iterator it = itsWriteHandleMap.begin(); it != itsWriteHandleMap.end(); it ++) {
    AsyncRequest* r = it->second;
    reqs[i] = r->mpiReq;
    i ++;
  }

  int res = MPI_Waitall(count, reqs, status);

  if (res != MPI_SUCCESS)
    THROW(CNProcException,"MPI_Waitall() failed: " << errorstr(res));

  for (std::map<int, AsyncRequest*>::const_iterator it = itsWriteHandleMap.begin(); it != itsWriteHandleMap.end(); it ++) {
    AsyncRequest* r = it->second;
    delete r;
  }

  itsWriteHandleMap.clear();
  itsCurrentWriteHandle = 0;
}


#endif // HAVE_MPI

} // namespace RTCPs
} // namespace LOFAR
