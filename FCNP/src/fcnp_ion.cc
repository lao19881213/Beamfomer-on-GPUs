#include <lofar_config.h>

#if defined HAVE_BGP_ION

#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Mutex.h>
#include <Common/LofarLogger.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>

#include <bpcore/bgp_collective_inlines.h>
#include <bpcore/bgp_atomic_ops.h>
#include <bpcore/ppc450_inlines.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <pthread.h>
#include <signal.h>

#include "fcnp_ion.h"
#include "protocol.h"

#define USE_SPIN_LOCKS
#undef USE_TIMER


namespace FCNP_ION {

using LOFAR::Semaphore;
using LOFAR::Mutex;


class Handshake {
  public:
    struct CnRequest {
      RequestPacket packet __attribute__ ((aligned(16)));
      Semaphore     slotFilled;

      CnRequest() : slotFilled(0) {}
    } cnRequest;

    struct IonRequest {
      size_t	      size;
      char	      *ptr;

#if 0
      pthread_mutex_t mutex;

      IonRequest()
      {
	pthread_mutex_init(&mutex, 0);
      }

      ~IonRequest()
      {
	pthread_mutex_destroy(&mutex);
      }
#endif
    } ionRequest;

    Semaphore writeFinished;

    Handshake() : writeFinished(0) {}
};

static Handshake		handshakes[MAX_CORES][MAX_CHANNELS][2] __attribute__ ((aligned(16))); // FIXME: variable size
static bool			useInterrupts;
static bool			initialized[256]; // FIXME
static std::vector<Handshake *> scheduledWriteRequests;
static uint32_t			vc0;
static int			fd;

#if defined USE_SPIN_LOCKS
static _BGP_Atomic		sendMutex = {0};
#else
static Mutex    		sendMutex;
#endif

static Mutex	        	scheduledRequestsLock;
static Mutex    		recvMutex;
static volatile bool		stop, stopped;

static _BGP_Atomic		nrMatchedWriteRequest = _BGP_ATOMIC_INIT(0);


static void setAffinity()
{
  cpu_set_t cpu_set;

  CPU_ZERO(&cpu_set);

  for (unsigned cpu = 1; cpu <= 3; cpu ++)
    CPU_SET(cpu, &cpu_set);

  if (sched_setaffinity(0, sizeof cpu_set, &cpu_set) != 0) {
    std::cerr << "WARNING: sched_setaffinity failed" << std::endl;
    perror("sched_setaffinity");
  }
}


static void raisePriority()
{
  struct sched_param sched_param;

  sched_param.sched_priority = sched_get_priority_max(SCHED_RR);

  if (pthread_setschedparam(pthread_self(), SCHED_RR, &sched_param) < 0)
    perror("pthread_setschedparam");
}


// Reading the tree status words seems to be expensive.  These wrappers
// minimize the number of status word reads.  Do not read/send packets
// without consulting these functions!

static inline void waitForFreeSendSlot()
{
#if 1
  _BGP_TreeFifoStatus stat;

  do
    stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
  while (stat.InjPyldCount > (_BGP_TREE_STATUS_MAX_PKTS - 1) * 16);
#else
  // only use this function while sendMutex locked!

  static unsigned minimumNumberOfFreeSendFIFOslots;

  while (minimumNumberOfFreeSendFIFOslots == 0) {
    _BGP_TreeFifoStatus stat;

    stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
    minimumNumberOfFreeSendFIFOslots = _BGP_TREE_STATUS_MAX_PKTS - std::max(stat.InjHdrCount, (stat.InjPyldCount + 15) / 16);
  }

  -- minimumNumberOfFreeSendFIFOslots;
#endif
}


#if 0
static unsigned minimumNumberOfFilledReceiveFIFOslots;
#endif


static inline void waitForIncomingPacket()
{
#if 1
  _BGP_TreeFifoStatus stat;

  do
    stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
  while (stat.RecPyldCount < 16);
#else
  while (minimumNumberOfFilledReceiveFIFOslots == 0) {
    _BGP_TreeFifoStatus stat;

    stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
    minimumNumberOfFilledReceiveFIFOslots = std::min(stat.RecHdrCount, stat.RecPyldCount / 16);
  }

  -- minimumNumberOfFilledReceiveFIFOslots;
#endif
}


static inline bool checkForIncomingPacket()
{
#if 1
  _BGP_TreeFifoStatus stat;

  stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
  return stat.RecPyldCount >= 16;
#else
  if (minimumNumberOfFilledReceiveFIFOslots == 0) {
    _BGP_TreeFifoStatus stat;

    stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
    minimumNumberOfFilledReceiveFIFOslots = std::min(stat.RecHdrCount, stat.RecPyldCount / 16);

    if (minimumNumberOfFilledReceiveFIFOslots == 0)
      return false;
  }

  -- minimumNumberOfFilledReceiveFIFOslots;
  return true;
#endif
}


inline static void copyPacket(RequestPacket *dst, const RequestPacket *src)
{
  unsigned sixteen;

  asm volatile (
    "lfpdx  0,0,%1;"
    "lfpdux 1,%1,%2;"
    "lfpdux 2,%1,%2;"
    "lfpdux 3,%1,%2;"
    "lfpdux 4,%1,%2;"
    "lfpdux 5,%1,%2;"
    "stfpdx 0,0,%0;"
    "stfpdux 1,%0,%2;"
    "stfpdux 2,%0,%2;"
    "lfpdux 6,%1,%2;"
    "lfpdux 7,%1,%2;"
    "lfpdux 0,%1,%2;"
    "stfpdux 3,%0,%2;"
    "stfpdux 4,%0,%2;"
    "stfpdux 5,%0,%2;"
    "lfpdux 1,%1,%2;"
    "lfpdux 2,%1,%2;"
    "lfpdux 3,%1,%2;"
    "stfpdux 6,%0,%2;"
    "stfpdux 7,%0,%2;"
    "stfpdux 0,%0,%2;"
    "lfpdux 4,%1,%2;"
    "lfpdux 5,%1,%2;"
    "lfpdux 6,%1,%2;"
    "stfpdux 1,%0,%2;"
    "stfpdux 2,%0,%2;"
    "stfpdux 3,%0,%2;"
    "lfpdux 7,%1,%2;"
    "stfpdux 4,%0,%2;"
    "stfpdux 5,%0,%2;"
    "stfpdux 6,%0,%2;"
    "stfpdux 7,%0,%2;"
    : "=b" (dst), "=b" (src), "=r" (sixteen)
    : "0" (dst), "1" (src), "2" (16)
    : "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7", "memory"
  );
}


inline static void handshakeComplete(Handshake *handshake)
{
  scheduledRequestsLock.lock();
  scheduledWriteRequests.push_back(handshake);
  scheduledRequestsLock.unlock();
}


static inline void lockSendFIFO()
{
#if defined USE_SPIN_LOCKS
  while (!_bgp_test_and_set(&sendMutex, 1))
    ;
#else
  sendMutex.lock();
#endif
}


static inline void unlockSendFIFO()
{
#if defined USE_SPIN_LOCKS
  _bgp_msync();
  sendMutex.atom = 0;
#else
  sendMutex.unlock();
#endif
}


static inline void sendPacketNoLocking(_BGP_TreePtpHdr *header, const void *ptr)
{
  waitForFreeSendSlot();
  _bgp_vcX_pkt_inject(&header->word, const_cast<void *>(ptr), vc0);
}


static inline void sendPacket(_BGP_TreePtpHdr *header, const void *ptr)
{
  lockSendFIFO();
  sendPacketNoLocking(header, ptr);
  unlockSendFIFO();
}


// Grabbing the sendMutex for each packet is too expensive on the ION.
// Provide a function that grabs one lock for 16 packets.

static inline void send16Packets(_BGP_TreePtpHdr *header, void *ptr)
{
  lockSendFIFO();

  for (char *p = (char *) ptr, *end = p + 16 * _BGP_TREE_PKT_MAX_BYTES; p < end; p += _BGP_TREE_PKT_MAX_BYTES) {    
    waitForFreeSendSlot();
    _bgp_vcX_pkt_inject(&header->word, p, vc0);
  }

  unlockSendFIFO();
}


static void sendAck(const RequestPacket *ack)
{
  _BGP_TreePtpHdr header;

#if 0
  header.Class	   = 0;
  header.Ptp	   = 1;
  header.Irq	   = 1;
  header.PtpTarget = ack->rank;
  header.CsumMode  = _BGP_TREE_CSUM_NONE;
#else
  header.word	   = (1 << 27) | (1 << 26) | (ack->rank << 2);
#endif

  sendPacket(&header, ack);
}


static void handleRequest(const RequestPacket *request)
{
  Handshake::CnRequest *cnRequest = &handshakes[request->rankInPSet][request->channel][request->type].cnRequest;

  //std::cout << "handleRequest: rank = " << request->rank << ", core = " << request->core << ", rankInPSet = " << request->rankInPSet << ", type = " << request->type << ", size = " << request->size << std::endl;

  if (request->type == RequestPacket::RESET) {
    if (!initialized[request->rankInPSet]) {
      initialized[request->rankInPSet] = true;
      sendAck(request);
    }
  } else {
    copyPacket(&cnRequest->packet, request); // TODO: avoid "large" memcpy
    cnRequest->slotFilled.up();
  }
}


static size_t handleReadRequest(RequestPacket *request, const char *ptr, size_t requestedSize)
{
  assert(requestedSize % 16 == 0 && request->size % 16 == 0);

#if defined USE_TIMER
  unsigned long long start_time = _bgp_GetTimeBase();
#endif

  size_t negotiatedSize = std::min(request->size, requestedSize);

  request->size = negotiatedSize;
  memcpy(request->messageHead, ptr, negotiatedSize % _BGP_TREE_PKT_MAX_BYTES);

  const char *end = ptr + negotiatedSize;
  ptr += negotiatedSize % _BGP_TREE_PKT_MAX_BYTES;

  sendAck(request);

  // now send the remaining data, which must be a multiple of the packet size

  _BGP_TreePtpHdr header;

#if 0
  header.Class	   = 0;
  header.Ptp	   = 1;
  header.Irq	   = 0;
  header.PtpTarget = request->rank;
  header.CsumMode  = _BGP_TREE_CSUM_NONE;
#else
  header.word	   = (1 << 27) | (request->rank << 2);
#endif

  for (; ptr < end - 15 * _BGP_TREE_PKT_MAX_BYTES; ptr += 16 * _BGP_TREE_PKT_MAX_BYTES)
    send16Packets(&header, (void *) ptr);

  lockSendFIFO();

  for (; ptr < end; ptr += _BGP_TREE_PKT_MAX_BYTES)
    sendPacketNoLocking(&header, (void *) ptr);

  unlockSendFIFO();

#if defined USE_TIMER
  unsigned long long stop_time = _bgp_GetTimeBase();
  std::cout << "read " << negotiatedSize << " bytes to " << request->rankInPSet << " @ " << (8 * negotiatedSize / ((stop_time - start_time) / 850e6) / 1e9) << " Gib/s" << std::endl;
#endif

  return negotiatedSize;
}


static size_t handleWriteRequest(RequestPacket *request, char *ptr, size_t requestedSize)
{
  assert(requestedSize % 16 == 0 && request->size % 16 == 0);

#if defined USE_TIMER
  unsigned long long start_time = _bgp_GetTimeBase();
#endif

  size_t negotiatedSize = std::min(request->size, requestedSize);

  request->size = negotiatedSize;
  memcpy(ptr, request->messageHead, negotiatedSize % _BGP_TREE_PKT_MAX_BYTES);

  const char *end = ptr + negotiatedSize;
  ptr += negotiatedSize % _BGP_TREE_PKT_MAX_BYTES;

  sendAck(request);

  // now receive the remaining data, which must be a multiple of the packet size

  while (ptr < end) {
    _BGP_TreePtpHdr header;

    waitForIncomingPacket();
    _bgp_vcX_pkt_receive(&header.word, ptr, vc0);

    if (header.Irq)
      handleRequest(reinterpret_cast<RequestPacket *>(ptr));
    else
      ptr += _BGP_TREE_PKT_MAX_BYTES;
  }

#if defined USE_TIMER
  unsigned long long stop_time = _bgp_GetTimeBase();
  std::cout << "write " << negotiatedSize << " bytes from " << request->rankInPSet << " @ " << (8 * negotiatedSize / ((stop_time - start_time) / 850e6) / 1e9) << " Gib/s" << std::endl;
#endif

  return negotiatedSize;
}


static void *pollThread(void *)
{
  setAffinity();
  raisePriority();

  _BGP_TreePtpHdr     header;
  _BGP_TreeFifoStatus stat;
  RequestPacket	      request __attribute__((aligned(16)));
  unsigned	      nrInterrupts = 0;

  while (!stop) {
    if (useInterrupts) {
      unsigned long long maxWaitTime = _bgp_GetTimeBase() + 50 * 850; // 50 us

      do
	stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
      while (stat.RecHdrCount == 0 && _bgp_GetTimeBase() < maxWaitTime);

      if (stat.RecHdrCount == 0) {
	int word;

	read(fd, &word, sizeof word); // wait for Irq packet
	++ nrInterrupts;
      }
    }

    if (_BGP_ATOMIC_READ((&nrMatchedWriteRequest)) == 0) {
      recvMutex.lock();

      if (checkForIncomingPacket()) {
	_bgp_vcX_pkt_receive(&header.word, &request, vc0);
	recvMutex.unlock();

	assert(header.Irq);
	handleRequest(&request);
      } else {
	recvMutex.unlock();
      }
    }
  }

  if (useInterrupts) {
    LOG_DEBUG_STR( "FCNP: Received " << nrInterrupts << " vc0 interrupts" );
    stopped = true;
  }

  return 0;
}


void IONtoCN_ZeroCopy(unsigned rankInPSet, unsigned channel, const void *ptr, size_t size)
{
  assert(size % 16 == 0 && (size_t) ptr % 16 == 0);
  assert(channel < MAX_CHANNELS);
  assert(rankInPSet < MAX_CORES);

  Handshake *handshake = &handshakes[rankInPSet][channel][RequestPacket::ZERO_COPY_READ];
  //pthread_mutex_lock(&handshake->ionRequest.mutex);

  while (size > 0) {
    handshake->cnRequest.slotFilled.down();

    // handle all read requests sequentially (and definitely those from multiple
    // cores from the same node!)
    static Mutex streamingSendMutex;

    streamingSendMutex.lock();
    size_t negotiatedSize = handleReadRequest(&handshake->cnRequest.packet, static_cast<const char *>(ptr), size);
    streamingSendMutex.unlock();

    size -= negotiatedSize;
    ptr = (const void *) ((const char *) ptr + negotiatedSize);
  }

  //pthread_mutex_unlock(&handshake->ionRequest.mutex);
}


void CNtoION_ZeroCopy(unsigned rankInPSet, unsigned channel, void *ptr, size_t size)
{
  assert(size % 16 == 0 && (size_t) ptr % 16 == 0);
  assert(channel < MAX_CHANNELS);
  assert(rankInPSet < MAX_CORES);

  Handshake *handshake = &handshakes[rankInPSet][channel][RequestPacket::ZERO_COPY_WRITE];
  //pthread_mutex_lock(&handshake->ionRequest.mutex);

  while (size > 0) {
    handshake->ionRequest.size = size;
    handshake->ionRequest.ptr  = static_cast<char *>(ptr);

    handshake->cnRequest.slotFilled.down();

    _bgp_fetch_and_add(&nrMatchedWriteRequest, 1);
    recvMutex.lock();
    size_t negotiatedSize = handleWriteRequest(&handshake->cnRequest.packet, handshake->ionRequest.ptr, handshake->ionRequest.size);
    recvMutex.unlock();
    _bgp_fetch_and_add(&nrMatchedWriteRequest, -1);

    size -= negotiatedSize;
    ptr = (void *) ((char *) ptr + negotiatedSize);
  }

  //pthread_mutex_unlock(&handshake->ionRequest.mutex);
}


#if 0
void writeUnaligned(unsigned rankInPSet, const void *ptr, size_t size)
{
  const char *src = static_cast<const char *>(ptr);

  while (size > 0) {
    size_t chunkSize = size % sizeof buffer;

    memcpy(buffer, src, chunkSize);
    src += chunkSize;
    size -= chunkSize;

    CNtoION_ZeroCopy(rankInPSet, buffer, chunkSize);
  }
}
#endif


static void openVC0()
{
  fd = open("/dev/tree0", O_RDWR);

  if (fd < 0) {
    perror("could not open /dev/tree0");
    exit(1);
  }

  if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
    perror("flock on /dev/tree0");
    exit(1);
  }

  vc0 = (uint32_t) mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (vc0 == (uint32_t) MAP_FAILED) {
    perror("could not mmap /dev/tree0");
    exit(1);
  }
}


static void drainFIFO()
{
  // check if previous run crashed halfway receiving a message

  _BGP_TreeFifoStatus stat;

  // Try to figure out how many quads are lingering around.  This cannot be
  // done 100% reliable (incoming packets do not increase RecHdrCount and
  // RecPyldCount atomically), so accept the answer if it is 16 times the same.

  int quadsToRead, previousQuadsToRead = -1;

  for (unsigned consistentAnswers = 0; consistentAnswers < 16;) {
    stat.status_word = _bgp_In32((uint32_t *) (vc0 + _BGP_TRx_Sx));
    quadsToRead = stat.RecPyldCount - 16 * stat.RecHdrCount;

    if (quadsToRead == previousQuadsToRead) {
      ++ consistentAnswers;
    } else {
      previousQuadsToRead = quadsToRead;
      consistentAnswers = 0;
    }
  }

  if (quadsToRead > 0)
    LOG_DEBUG_STR( "FCNP: Dropped " << quadsToRead << " lingering quadwords from packets of a previous job" );

  while (-- quadsToRead >= 0)
    _bgp_QuadLoad(vc0 + _BGP_TRx_Sx, 0);

  // check if previous run crashed halfway sending a message

  if (stat.InjPyldCount % 16 != 0) {
    // TODO: recover from this
    std::cerr << "previous run crashed while sending a message -- please reboot partition" << std::endl;
    exit(1);
  }

  // drain lingering packets from previous jobs

  uint64_t time    = _bgp_GetTimeBase() + 850000000;
  unsigned dropped = 0;

  while (_bgp_GetTimeBase() < time)
    if (checkForIncomingPacket()) {
      _BGP_TreePtpHdr header;
      _bgp_vcX_hdr_receive(&header.word, vc0);
      _bgp_vcX_pkt_receiveNoHdrNoStore(0, vc0); // drop everything
      ++ dropped;
    }

  if (dropped > 0)
    LOG_DEBUG_STR( "FCNP: Dropped " << dropped << " lingering packets from previous job" );
}


static pthread_t thread;


static void sigHandler(int)
{
}


void init(bool enableInterrupts)
{
  if (enableInterrupts) {
    struct sigaction sa;

    sigemptyset(&sa.sa_mask);
    sa.sa_flags   = 0;
    sa.sa_handler = sigHandler;

    if (sigaction(SIGUSR1, &sa, 0) != 0)
      perror("sigaction");
  }

  useInterrupts = enableInterrupts;

  openVC0();
  drainFIFO();

  if (pthread_create(&thread, 0, pollThread, 0) != 0) {
    perror("pthread_create");
    exit(1);
  }
}


void end()
{
  stop = true;

  if (useInterrupts)
    while (!stopped) {
      if (pthread_kill(thread, SIGUSR1) != 0)
	perror("pthread_kill");

      usleep(25000);
    }

  if (pthread_join(thread, 0) != 0) {
    perror("pthread_join");
    exit(1);
  }

  close(vc0);
}

} // namespace FCNP_ION

#endif // defined HAVE_BGP_ION
