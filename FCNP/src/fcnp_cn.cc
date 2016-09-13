#include <lofar_config.h>

#if defined HAVE_BGP_CN

#include <Common/LofarLogger.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <iostream>

#include <bpcore/bgp_collective_inlines.h>
#include <common/bgp_personality_inlines.h>
#include <spi/kernel_interface.h>
#include <spi/lockbox_interface.h>

#include "fcnp_cn.h"
#include "protocol.h"


namespace FCNP_CN {

static unsigned nextMutex = LOCKBOX_RESERVED_LOCKID; // TODO: deallocate mutexes


class Semaphore
{
  public:
    Semaphore(unsigned value = 0)
    {
      if (LockBox_AllocateCounter(-- nextMutex, &level, 0, 4, 0) < 0) {
	LOG_FATAL("Could not allocate lockbox");
	exit(1);
      }

      LockBox_Write(level, value);
    }

    ~Semaphore()
    {
      // TODO
    }

    void up()
    {
      LockBox_FetchAndInc(level);
    }

    void down()
    {
      while (LockBox_FetchAndDec(level) == 0)
	;
    }

    bool tryDown()
    {
      return LockBox_FetchAndDec(level) != 0;
    }

  private:
    LockBox_Counter_t level;
};



static _BGP_Personality_t personality;
static LockBox_Mutex_t	  sendMutex, stateMutex;
static Semaphore	  shmEmpty(1);
static _BGP_TreeHwHdr	  requestHeader, dataHeader;
static unsigned		  myCore, myRankInPSet;
static int		  shmFD;

static struct Shm {
  RequestPacket	reply; // keep 16-byte aligned
  volatile unsigned	minimumNumberOfFreeSendFIFOslots;
  volatile unsigned	minimumNumberOfFilledReceiveFIFOslots;
  volatile bool		recvFifoLocked, replyAvailable[4];
  volatile bool		initialized;
} *shm;
void *unalignedShmPtr;


// Reading the tree status words seems to be expensive.  These wrappers
// minimize the number of status word reads.  Do not read/send packets
// without consulting these functions!

static inline void waitForFreeSendSlot()
{
#if 1
  _BGP_TreeFifoStatus stat;

  do
    _bgp_TreeGetStatusVC0(&stat);
  while (stat.InjPyldCount > (_BGP_TREE_STATUS_MAX_PKTS - 1) * 16);
#else
  // only use this function while sendMutex locked!
  unsigned slotsFree = shm->minimumNumberOfFreeSendFIFOslots;

  while (slotsFree == 0) {
    _BGP_TreeFifoStatus stat;

    _bgp_TreeGetStatusVC0(&stat);
    slotsFree = _BGP_TREE_STATUS_MAX_PKTS - std::max(stat.InjHdrCount, (stat.InjPyldCount + 15) / 16);
  }
  shm->minimumNumberOfFreeSendFIFOslots = slotsFree - 1;
#endif
}


static inline bool checkForIncomingPacket()
{
#if 1
  _BGP_TreeFifoStatus stat;

  _bgp_TreeGetStatusVC0(&stat);
  return stat.RecPyldCount >= 16;
#else
  // only use this function while recvMutex locked!
  unsigned slotsFilled = shm->minimumNumberOfFilledReceiveFIFOslots;

  if (slotsFilled == 0) {
    _BGP_TreeFifoStatus stat;

    _bgp_TreeGetStatusVC0(&stat);
    slotsFilled = std::min(stat.RecHdrCount, stat.RecPyldCount / 16);

    if (slotsFilled == 0)
      return false;
  }

  shm->minimumNumberOfFilledReceiveFIFOslots = slotsFilled - 1;
  return true;
#endif
}


static inline void waitForIncomingPacket()
{
  // only use this function while recvMutex locked!

  while (!checkForIncomingPacket())
    ;
}


static void sendRequest(/*const*/ RequestPacket *request)
{
  LockBox_MutexLock(sendMutex);
  waitForFreeSendSlot();
  _bgp_TreeRawSendPacketVC0(&requestHeader, request);
  LockBox_MutexUnlock(sendMutex);
}


static void waitForReply()
{
//std::clog << "before lock(stateMutex) A" << std::endl;
  LockBox_MutexLock(stateMutex);
//std::clog << "passed lock(stateMutex) A, replyAvailable[" << myCore << "] = " << shm->replyAvailable[myCore] << ", recvFifoLocked = " << shm->recvFifoLocked << std::endl;

  while (!shm->replyAvailable[myCore]) {
    if (!shm->recvFifoLocked) {
      shm->recvFifoLocked = true;

      do {
	LockBox_MutexUnlock(stateMutex);

//std::clog << "before shmEmpty.down() B" << std::endl;
	shmEmpty.down();
//std::clog << "passed shmEmpty.down() B" << std::endl;

	do {
	  waitForIncomingPacket();

	  _BGP_TreeHwHdr replyHeader;
	  _bgp_TreeRawReceivePacketVC0(&replyHeader, &shm->reply);

	  assert(replyHeader.PtpHdr.Irq);
	  assert(shm->reply.rank == personality.Network_Config.Rank);
if (shm->reply.type == RequestPacket::RESET) std::clog << "ignored reset ack" << std::endl;
	} while (shm->reply.type == RequestPacket::RESET); // ignore

//std::clog << "before lock(stateMutex) C, shm->core = " << shm->reply.core << ", shm->type = " << shm->reply.type << std::endl;
	LockBox_MutexLock(stateMutex);
//std::clog << "passed lock(stateMutex) C" << std::endl;
	shm->replyAvailable[shm->reply.core] = true;
      } while (shm->reply.core != myCore && shm->reply.type == RequestPacket::ZERO_COPY_WRITE);

      if (shm->reply.core == myCore && shm->reply.type == RequestPacket::ZERO_COPY_WRITE)
	shm->recvFifoLocked = false;
    } else {
      LockBox_MutexUnlock(stateMutex);

      for (uint64_t time = _bgp_GetTimeBase() + 1000; _bgp_GetTimeBase() < time;)
	;
      
      LockBox_MutexLock(stateMutex);
    }
  }

  assert(shm->reply.core == myCore);
  shm->replyAvailable[myCore] = false; // reset for next request
  LockBox_MutexUnlock(stateMutex);
//std::clog << "received reply" << std::endl;
}


static void receiveData(char *ptr)
{
  unsigned firstBytes = shm->reply.size % _BGP_TREE_PKT_MAX_BYTES;
  memcpy(ptr, shm->reply.messageHead, firstBytes);
  ptr += firstBytes;
  char *end = ptr + (shm->reply.size & ~(_BGP_TREE_PKT_MAX_BYTES - 1));

  shmEmpty.up();

  // now receive the remaining data, which must be a multiple of the packet size
  assert((end - ptr) % _BGP_TREE_PKT_MAX_BYTES == 0);

  while (ptr < end) {
    _BGP_TreeHwHdr replyHeader;

    waitForIncomingPacket();
    _bgp_TreeRawReceivePacketVC0(&replyHeader, ptr);

    if (replyHeader.PtpHdr.Irq) {
      if (reinterpret_cast<RequestPacket *>(ptr)->type == RequestPacket::RESET) {
std::clog << "ignored reset ack (2)" << std::endl;
	continue; // ignore
      }

//std::clog << "before shmEmpty.down() D" << std::endl;
      shmEmpty.down();
//std::clog << "passed shmEmpty.down() D" << std::endl;
      memcpy(&shm->reply, ptr, _BGP_TREE_PKT_MAX_BYTES);

      if (shm->reply.rank != personality.Network_Config.Rank)
	std::clog << "BAD PACKET: type = " << shm->reply.type << ", rank = " << shm->reply.rank << ", core = " << shm->reply.core << ", rankInPSet = " << shm->reply.rankInPSet << ", size = " << shm->reply.size << "; my rank = " << personality.Network_Config.Rank << std::endl;

      assert(shm->reply.rank == personality.Network_Config.Rank);
      assert(shm->reply.core != myCore);
      assert(shm->reply.type == RequestPacket::ZERO_COPY_WRITE); // assured by ION

      LockBox_MutexLock(stateMutex);
      shm->replyAvailable[shm->reply.core] = true;
      LockBox_MutexUnlock(stateMutex);
    } else {
      ptr += _BGP_TREE_PKT_MAX_BYTES;
    }
  }

  LockBox_MutexLock(stateMutex);
  shm->recvFifoLocked = false;
  LockBox_MutexUnlock(stateMutex);
}


void IONtoCN_ZeroCopy(unsigned channel, void *ptr, size_t size)
{
  //std::cout << "IONtoCN_ZeroCopy(" << ptr << ", " << size << ")" << std::endl;

  assert(size % 16 == 0 && (size_t) ptr % 16 == 0);
  assert(channel < MAX_CHANNELS);

  char *dst = static_cast<char *>(ptr);

  while (size > 0) {
    RequestPacket request __attribute__ ((aligned(16)));

    request.type	= RequestPacket::ZERO_COPY_READ;
    request.rank	= personality.Network_Config.Rank;
    request.core	= myCore;
    request.rankInPSet	= myRankInPSet;
    request.channel	= channel;
    request.size	= size;

    sendRequest(&request);
    waitForReply();
    assert(shm->reply.type == RequestPacket::ZERO_COPY_READ);

    size_t ackdSize = shm->reply.size; // ION may ack fewer bytes than requested
    //std::cout << "read: " << ackdSize << " bytes" << std::endl;
    receiveData(dst); // may not read shm->reply after this call
    size -= ackdSize;
    dst  += ackdSize;
  }
}


static void sendData(const char *ptr, size_t size)
{
  assert(size % _BGP_TREE_PKT_MAX_BYTES == 0);
  //LockBox_MutexUnlock(shmMutex);
  shmEmpty.up();

  for (const char *end = ptr + size; ptr < end;) {
    // TODO: do not grab mutex every time
    LockBox_MutexLock(sendMutex);
    waitForFreeSendSlot();
    _bgp_TreeRawSendPacketVC0(&dataHeader, const_cast<char *>(ptr));
    ptr += _BGP_TREE_PKT_MAX_BYTES;
    LockBox_MutexUnlock(sendMutex);
  }
}


void CNtoION_ZeroCopy(unsigned channel, const void *ptr, size_t size)
{
  //std::cout << "CNtoION_ZeroCopy(" << ptr << ", " << size << ")" << std::endl;

  assert(size % 16 == 0 && (size_t) ptr % 16 == 0);
  assert(channel < MAX_CHANNELS);

  const char *src = static_cast<const char *>(ptr);

  while (size > 0) {
    RequestPacket request __attribute__ ((aligned(16)));

    request.type	= RequestPacket::ZERO_COPY_WRITE;
    request.rank	= personality.Network_Config.Rank;
    request.core	= myCore;
    request.rankInPSet	= myRankInPSet;
    request.channel	= channel;
    request.size	= size;

    memcpy(request.messageHead, src, std::min(size, sizeof request.messageHead));
    sendRequest(&request);
    waitForReply();
    assert(shm->reply.type == RequestPacket::ZERO_COPY_WRITE);

    size_t ackdSize = shm->reply.size; // ION may ack fewer bytes than requested
    //std::cout << "write: " << ackdSize << " bytes" << std::endl;

    sendData(src + ackdSize % _BGP_TREE_PKT_MAX_BYTES, ackdSize & ~(_BGP_TREE_PKT_MAX_BYTES - 1));
    size -= ackdSize;
    src += ackdSize;
  }
}


static void getPersonality()
{
  if (Kernel_GetPersonality(&personality, sizeof personality) != 0) {
    std::cerr << "could not get personality" << std::endl;
    exit(1);
  }
}


static void openShm()
{
  size_t shmSize = (sizeof *shm + 15) & ~15;

  if (Kernel_ProcessCount() > 1) {
    if ((shmFD = shm_open("pkt", O_RDWR, 0600)) < 0) {
      perror("shm_open");
      exit(1);
    }

    if (ftruncate(shmFD, shmSize) < 0) {
      perror("ftruncate shm");
      exit(1);
    }

    if ((unalignedShmPtr = mmap(0, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFD, 0)) == MAP_FAILED) {
      perror("mmap shm");
      exit(1);
    }

  } else {
    unalignedShmPtr = new char[shmSize];
  }

  shm = reinterpret_cast<Shm *>((reinterpret_cast<size_t>(unalignedShmPtr) + 15 ) & ~15);
  //shm->minimumNumberOfFreeSendFIFOslots      = 0; // force reevaluation
  //shm->minimumNumberOfFilledReceiveFIFOslots = 0;
}

#if 0
static void closeShm()
{
  if (Kernel_ProcessCount() > 1) {
    if (munmap(shm, (sizeof *shm + 15) & ~15) < 0) {
      perror("munmap shm");
      exit(1);
    }

    close(shmFD);

    if (shm_unlink("pkt") < 0) {
      perror("shm_unlink");
      exit(1);
    }
  } else {
    delete [] static_cast<char *>(unalignedShmPtr);
  }
}
#endif

static void allocateMutexes()
{
  if (LockBox_AllocateMutex(-- nextMutex, &sendMutex, 0, 4, LOCKBOX_ORDERED_ALLOC) < 0 ||
      LockBox_AllocateMutex(-- nextMutex, &stateMutex, 0, 4, LOCKBOX_ORDERED_ALLOC) < 0) {
    std::cerr << "Could not allocate lockbox" << std::endl;
    exit(1);
  }
}


static void initHeaders()
{
  requestHeader.PtpHdr.Class	 = 0;
  requestHeader.PtpHdr.Ptp	 = 1;
  requestHeader.PtpHdr.Irq	 = 1;
  requestHeader.PtpHdr.PtpTarget = BGP_Personality_treeIONodeP2PAddr(&personality);
  requestHeader.PtpHdr.CsumMode  = _BGP_TREE_CSUM_NONE;

  dataHeader			 = requestHeader;
  dataHeader.PtpHdr.Irq		 = 0;
}


static void getConfiguration()
{
  unsigned xPos = personality.Network_Config.Xcoord;
  unsigned yPos = personality.Network_Config.Ycoord;
  unsigned zPos = personality.Network_Config.Zcoord;

  unsigned xPsetSize, yPsetSize, zPsetSize;
  unsigned psetSize = personality.Network_Config.PSetSize;

  switch (psetSize) {
    case 16 :	xPsetSize = 4, yPsetSize = 2, zPsetSize = 2;
		break;

    case 32 :	xPsetSize = 4, yPsetSize = 4, zPsetSize = 2;
		break;

    case 64 :	xPsetSize = 4, yPsetSize = 4, zPsetSize = 4;
		break;

    case 128 :	xPsetSize = 4, yPsetSize = 4, zPsetSize = 8;
		break;

    case 256 :	xPsetSize = 8, yPsetSize = 4, zPsetSize = 8;
		break;

    case 512 :	xPsetSize = 8, yPsetSize = 8, zPsetSize = 8;
		break;

    default :	std::cerr << "FCNP: cannot determine PSet dimensions" << std::endl;
		exit(1);
  }

  unsigned xPsetPos = xPos % xPsetSize;
  unsigned yPsetPos = yPos % yPsetSize;
  unsigned zPsetPos = zPos % zPsetSize;

  unsigned pos = ((zPsetPos * yPsetSize + yPsetPos) * xPsetSize) + xPsetPos;

  myCore       = Kernel_PhysicalProcessorID();
  myRankInPSet = psetSize * myCore + pos;
}


static void drainFIFO()
{
  // check if previous run crashed halfway sending a message

  _BGP_TreeFifoStatus stat;

  _bgp_TreeGetStatusVC0(&stat);

  if (stat.RecPyldCount % 16 != 0 || stat.InjPyldCount % 16 != 0) {
    // TODO: recover from this
    std::cerr << "previous run crashed while sending or receiving a message -- please reboot partition" << std::endl;
    exit(1);
  }

  // drain lingering packets from previous jobs

  LockBox_MutexLock(stateMutex);

  if (!shm->initialized) {
    shm->initialized = true; // only one core needs to drain the FIFO

    _BGP_TreeHwHdr header;
    uint64_t time    = _bgp_GetTimeBase() + 850000000;
    unsigned dropped = 0;

    while (_bgp_GetTimeBase() < time)
      if (checkForIncomingPacket()) {
	_bgp_TreeRawReceiveHeader(0, &header);
	_bgp_TreeRawReceivePacketNoHdrNoStore(0); // drop everything
	++ dropped;
      }

    if (dropped > 0)
      std::clog << "dropped " << dropped << " lingering packets from previous job" << std::endl;

    // now send RESET request
    RequestPacket request __attribute__ ((aligned(16)));

    request.type	= RequestPacket::RESET;
    request.rank	= personality.Network_Config.Rank;
    request.rankInPSet	= myRankInPSet;

    do {
      sendRequest(&request); // may even block FIFO; ION will eventually drain it
      usleep(250000);
    } while (!checkForIncomingPacket());

    // wait for RESET reply
    _bgp_TreeRawReceivePacketVC0(&header, &request); // reuse space for reply

    assert(header.PtpHdr.Irq);
    assert(request.type == RequestPacket::RESET);
  }

  LockBox_MutexUnlock(stateMutex);
}


void init()
{
  getPersonality();
  openShm();
  getConfiguration();
  initHeaders();
  allocateMutexes();
  drainFIFO();

#if 0
  //std::cout << "RankInPSet = " << personality.Network_Config.RankInPSet;
  std::cout << "RankInPSet = " << myRankInPSet;
  std::cout << ", PSetNum = " << personality.Network_Config.PSetNum;
  std::cout << ", Rank = " << personality.Network_Config.Rank;
  std::cout << ", Kernel_PhysicalProcessorID = " << Kernel_PhysicalProcessorID() << std::endl;
  exit(0);
#endif
}

} // namespace FCNP_CN

#endif // defined HAVE_BGP_CN
