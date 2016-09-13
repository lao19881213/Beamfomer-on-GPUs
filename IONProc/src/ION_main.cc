//#  ION_main.cc:
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
//#  $Id: ION_main.cc 27510 2013-11-25 21:19:42Z schoenmakers $

#include <lofar_config.h>
#include <GlobalVars.h>

#include <CommandServer.h>
#include <Common/LofarLogger.h>
#include <Common/CasaLogSink.h>
#include <Common/NewHandler.h>
#include <Common/SystemCallException.h>
#include <Interface/CN_Command.h>
#include <Interface/CN_Mapping.h>
#include <Interface/Exceptions.h>
#include <Interface/SmartPtr.h>
#include <Interface/Stream.h>
#include <Interface/Parset.h>
#include <ION_Allocator.h>
#include <Delays.h>
#include <SSH.h>
#include <Stream/SocketStream.h>
#include <StreamMultiplexer.h>
#include <IONProc/Package__Version.h>

#include <boost/multi_array.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#if defined HAVE_MPI
#include <mpi.h>
#endif

#if defined HAVE_FCNP && defined __PPC__ && !defined USE_VALGRIND
#include <FCNP/fcnp_ion.h>
#include <FCNP_ServerStream.h>
#endif

#ifdef USE_VALGRIND
extern "C" {
#include <valgrind/valgrind.h>

/*
 * Valgrind wrappers to replace functions which use Double Hummer instructions,
 * since valgrind can't cope with them.
 *
 * Outside valgrind, these functions are not used.
 */

void *I_WRAP_SONAME_FNNAME_ZZ(Za,memcpy)( void *b, const void *a, size_t n) {
    char *s1 = static_cast<char*>(b);
    const char *s2 = static_cast<const char*>(a);
    for(; 0<n; --n)*s1++ = *s2++;
    return b;
}

void *I_WRAP_SONAME_FNNAME_ZZ(Za,memset)( void *dest, int val, size_t len) {
  unsigned char *ptr = static_cast<unsigned char*>(dest);
  while (len-- > 0)
    *ptr++ = val;
  return dest;
}

}
#endif

// install a new handler to produce backtraces for std::bad_alloc
LOFAR::NewHandler h(LOFAR::BadAllocException::newHandler);

// Use a terminate handler that can produce a backtrace.
LOFAR::Exception::TerminateHandler t(LOFAR::Exception::terminate);


namespace LOFAR {
namespace RTCP {

static boost::multi_array<char, 2>	  ipAddresses;

#if defined HAVE_FCNP && defined __PPC__ && !defined USE_VALGRIND
static struct InitFCNP {
  InitFCNP() { FCNP_ION::init(true); }
  ~InitFCNP() { FCNP_ION::end(); }
} initFCNP;
#endif

static void createAllCNstreams()
{
  LOG_DEBUG_STR("Create streams to CN nodes ...");

  const char *streamType = getenv("CN_STREAM_TYPE");

  if (streamType != 0)
    cnStreamType = streamType;
  else
#if !defined HAVE_BGP_ION
    cnStreamType = "NULL";
#elif defined HAVE_FCNP && defined __PPC__ && !defined USE_VALGRIND
    cnStreamType = "FCNP";
#else
    cnStreamType = "TCPKEY";
#endif

  allCNstreams.resize(nrPsets,nrCNcoresInPset);

#ifdef CLUSTER_SCHEDULING
  for (unsigned pset = 0; pset < nrPsets; pset ++)
    for (unsigned core = 0; core < nrCNcoresInPset; core ++)
      allCNstreams[pset][core] = createCNstream(pset, core, 0);
#else
  for (unsigned core = 0; core < nrCNcoresInPset; core ++)
    allCNstreams[myPsetNumber][core] = createCNstream(myPsetNumber, core, 0);
#endif

  LOG_DEBUG_STR("Create streams to CN nodes done");
}


static void stopCNs()
{
  LOG_DEBUG_STR("Stopping " << nrCNcoresInPset << " cores ...");

  CN_Command command(CN_Command::STOP);

  for (unsigned core = 0; core < nrCNcoresInPset; core ++)
    command.write(allCNstreams[myPsetNumber][core]);

  LOG_DEBUG_STR("Stopping " << nrCNcoresInPset << " cores: done");
}


static void createAllIONstreams()
{
  LOG_DEBUG_STR("Create streams between I/O nodes ...");

  if (myPsetNumber == 0) {
    allIONstreams.resize(nrPsets);
    allIONstreamMultiplexers.resize(nrPsets);

    for (unsigned ion = 1; ion < nrPsets; ion ++) {
      allIONstreams[ion] = new SocketStream(ipAddresses[ion].origin(), 4000 + ion, SocketStream::TCP, SocketStream::Client);
      allIONstreamMultiplexers[ion] = new StreamMultiplexer(*allIONstreams[ion]);
      allIONstreamMultiplexers[ion]->start();
    }
  } else {
    allIONstreams.push_back(new SocketStream(ipAddresses[myPsetNumber].origin(), 4000 + myPsetNumber, SocketStream::TCP, SocketStream::Server));
    allIONstreamMultiplexers.push_back(new StreamMultiplexer(*allIONstreams[0]));
    allIONstreamMultiplexers[0]->start();
  }

  LOG_DEBUG_STR("Create streams between I/O nodes: done");
}


static void enableCoreDumps()
{
  struct rlimit rlimit;

  rlimit.rlim_cur = RLIM_INFINITY;
  rlimit.rlim_max = RLIM_INFINITY;

  if (setrlimit(RLIMIT_CORE, &rlimit) < 0)
    perror("warning: setrlimit on unlimited core size failed");

#if defined HAVE_BGP
  if (system("echo /tmp/%e.core >/proc/sys/kernel/core_pattern") < 0)
    LOG_WARN("Could not change /proc/sys/kernel/core_pattern");
#endif

  LOG_DEBUG("Coredumps enabled");
}


static void abortHandler(int sig)
{
  (void)sig;

  abort();
}


static void installSigHandlers()
{
  // ignore SIGPIPE
  if (signal(SIGPIPE, SIG_IGN) == SIG_ERR)
    perror("warning: ignoring SIGPIPE failed");

  // force abort() on a few signals, as OpenMPI appears to be broken in this regard
  if (signal(SIGBUS, abortHandler) == SIG_ERR)
    perror("warning: rerouting SIGBUS failed");
  if (signal(SIGSEGV, abortHandler) == SIG_ERR)
    perror("warning: rerouting SIGSEGV failed");
  if (signal(SIGILL, abortHandler) == SIG_ERR)
    perror("warning: rerouting SIGILL failed");
  if (signal(SIGFPE, abortHandler) == SIG_ERR)
    perror("warning: rerouting SIGFPE failed");
}


#if defined FLAT_MEMORY

static void   *flatMemoryAddress = reinterpret_cast<void *>(0x50000000);
static size_t flatMemorySize     = 1536 * 1024 * 1024;

static void mmapFlatMemory()
{
  // mmap a fixed area of flat memory space to increase performance. 
  // currently only 1.5 GiB can be allocated, we mmap() the maximum
  // available amount
  int fd = open("/dev/flatmem", O_RDONLY);

  if (fd < 0) { 
    perror("open(\"/dev/flatmem\", O_RDONLY)");
    exit(1);
  }
 
  if (mmap(flatMemoryAddress, flatMemorySize, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) {
    perror("mmap flat memory");
    exit(1);
  } 

  close(fd);

  LOG_DEBUG_STR("Mapped " << flatMemorySize << " bytes of fast memory at " << flatMemoryAddress);
}

static void unmapFlatMemory()
{
  if (munmap(flatMemoryAddress, flatMemorySize) < 0)
    perror("munmap flat memory");
}

#endif


static void master_thread()
{
#if !defined HAVE_PKVERSION
  std::string type = "brief";
  Version::show<IONProcVersion> (std::clog, "IONProc", type);
#endif  
  
  LOG_DEBUG("Master thread running");

  enableCoreDumps();
  installSigHandlers();

  try {

#if defined FLAT_MEMORY
    mmapFlatMemory();
#endif

    if (getenv("AIPSPATH") == 0)
      setenv("AIPSPATH", "/globalhome/lofarsystem/packages/root/bgp_ion/", 0);

#if defined HAVE_BGP
    // TODO: how to figure these out?
    nrCNcoresInPset = 64;

    // nrPsets is communicated by MPI
#else
    const char *nr_psets  = getenv("NR_PSETS");
    const char *pset_size = getenv("PSET_SIZE");

    if (nr_psets == 0)
      throw IONProcException("environment variable NR_PSETS must be defined", THROW_ARGS);

    if (pset_size == 0)
      throw IONProcException("environment variable PSET_SIZE must be defined", THROW_ARGS);

    nrPsets = boost::lexical_cast<unsigned>(nr_psets);
    nrCNcoresInPset = boost::lexical_cast<unsigned>(pset_size);
#endif

    createAllCNstreams();
    createAllIONstreams();
    { CommandServer s; s.start(); }

    stopCNs();

#if defined FLAT_MEMORY
    unmapFlatMemory();
#endif

  } catch (Exception &ex) {
    LOG_FATAL_STR("Master thread caught Exception: " << ex);
  } catch (std::exception &ex) {
    LOG_FATAL_STR("Master thread caught std::exception: " << ex.what());
  } catch (...) {
    LOG_FATAL("Master thread caught non-std::exception: ");
  }

  LOG_DEBUG("Master thread stopped");
}


} // namespace RTCP
} // namespace LOFAR


int main(int argc, char **argv)
{
  using namespace LOFAR;
  using namespace LOFAR::RTCP;

#if defined HAVE_MPI
#if 1
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed" << std::endl;
    exit(1);
  }
#else
  int provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI does not provide MPI_THREAD_MULTIPLE" << std::endl;
    exit(1);
  }
#endif

  MPI_Comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&myPsetNumber));
  MPI_Comm_size(MPI_COMM_WORLD, reinterpret_cast<int *>(&nrPsets));
#else
  (void) argc;
  (void) argv;
#endif

#if defined HAVE_MPI
  ipAddresses.resize(boost::extents[nrPsets][16]);

#if defined HAVE_BGP_ION
  ParameterSet personality("/proc/personality.sh");
#if 0
  unsigned realPsetNumber = personality.getUint32("BG_PSETNUM");

  if (myPsetNumber != realPsetNumber) {
    std::cerr << "myPsetNumber (" << myPsetNumber << ") != realPsetNumber (" << realPsetNumber << ')' << std::endl;
    exit(1);
  }
#endif

  std::string myIPaddress = personality.getString("BG_IP");
  strcpy(ipAddresses[myPsetNumber].origin(), myIPaddress.c_str());
#else
  const char *uri = getenv("OMPI_MCA_orte_local_daemon_uri");

  if (uri == 0) {
    std::cerr << "\"OMPI_MCA_orte_local_daemon_uri\" not in environment" << std::endl;
    exit(1);
  }

  if (sscanf(uri, "%*u.%*u;tcp://%[0-9.]:%*u", ipAddresses[myPsetNumber].origin()) != 1) {
    std::cerr << "could not parse environment variable \"OMPI_MCA_orte_local_daemon_uri\"" << std::endl;
    exit(1);
  }
#endif

  for (unsigned root = 0; root < nrPsets; root ++)
    if (MPI_Bcast(ipAddresses[root].origin(), sizeof(char [16]), MPI_CHAR, root, MPI_COMM_WORLD) != MPI_SUCCESS) {
      std::cerr << "MPI_Bcast failed" << std::endl;
      exit(1);
    }
#endif
  
#if defined HAVE_BGP
  INIT_LOGGER_WITH_SYSINFO(str(boost::format("IONProc@%02d") % myPsetNumber));
  bool isProduction = argc > 1 && argv[1][0] == '1';
  LOGCOUT_SETLEVEL(isProduction ? 4 : 8); // do (not) show debug info
#elif defined HAVE_LOG4CPLUS
  // do nothing
#elif defined HAVE_LOG4CXX
  Context::initialize();
  setLevel("Global", 8);
#else
  INIT_LOGGER_WITH_SYSINFO(str(boost::format("IONProc@%02d") % myPsetNumber));
#endif

  if (!SSH_Init()) {
    std::cerr << "SSH subsystem init failed" << std::endl;
    exit(1);
  }

  if (!Casacore_Init()) {
    std::cerr << "Casacore subsystem init failed" << std::endl;
    exit(1);
  }

  //CasaLogSink::attach();

  master_thread();

  SSH_Finalize();

#if defined HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
