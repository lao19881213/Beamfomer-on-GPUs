//#  CN_Processing_main.cc:
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
//#  $Id: CN_Processing_main.cc 21558 2012-07-12 09:35:39Z loose $

#include <lofar_config.h>

#include <CNProc/LocationInfo.h>
#include <CNProc/CN_Processing.h>
#include <CNProc/Package__Version.h>
#include <Common/Exception.h>
#include <Common/LofarLogger.h>
#include <Common/NewHandler.h>
#include <Interface/Allocator.h>
#include <Interface/CN_Command.h>
#include <Interface/Exceptions.h>
#include <Interface/Parset.h>
#include <Interface/SmartPtr.h>
#include <Interface/Stream.h>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <execinfo.h>

#if defined CLUSTER_SCHEDULING
#define LOG_CONDITION 1
#else
#define LOG_CONDITION (locationInfo.rankInPset() == 0)
#endif

#if defined HAVE_MPI
#define MPICH_IGNORE_CXX_SEEK
#include <mpi.h>
#endif

#if defined HAVE_FCNP && defined HAVE_BGP_CN && !defined USE_VALGRIND
#include <FCNP_ClientStream.h>
#include <FCNP/fcnp_cn.h>
#endif

#include <cstdio>
#include <cstring>

#if defined HAVE_BGP && defined HAVE_FFTW2
// use our own memory managment to both use new/delete and
// to avoid fftw from calling exit() when there is not
// enough memory.

// We can't redirect the malloc()s done by fftw3 yet as they are hard-coded.
// Be warned that fftw also abort()s or exit()s when malloc fails.
#define REROUTE_FFTW2_MALLOC
#endif

#if defined REROUTE_FFTW2_MALLOC
#include <fftw.h>
#endif

// install a new handler to produce backtraces for std::bad_alloc
LOFAR::NewHandler h(LOFAR::BadAllocException::newHandler);

using namespace LOFAR;
using namespace LOFAR::RTCP;

// Use a terminate handler that can produce a backtrace.
Exception::TerminateHandler t(Exception::terminate);

static const char *ionStreamType;


static void getIONstreamType()
{
  if ((ionStreamType = getenv("CN_STREAM_TYPE")) == 0)
#if !defined HAVE_BGP_CN
    ionStreamType = "NULL";
#elif defined HAVE_FCNP && defined __PPC__ && !defined USE_VALGRIND
    ionStreamType = "FCNP";
#else
    ionStreamType = "TCPKEY";
#endif

#if defined HAVE_FCNP && defined HAVE_BGP_CN && !defined USE_VALGRIND
  if (ionStreamType == "FCNP")
    FCNP_CN::init();
#endif
}


static Stream *createIONstream(unsigned channel, const LocationInfo &locationInfo)
{
#if defined HAVE_FCNP && defined HAVE_BGP_CN && !defined USE_VALGRIND
  if (strcmp(ionStreamType, "FCNP") == 0)
    return new FCNP_ClientStream(channel);
#endif

  unsigned nrPsets = locationInfo.nrPsets();
  unsigned psetSize = locationInfo.psetSize();
  unsigned psetNumber = locationInfo.psetNumber();
  unsigned rankInPset = locationInfo.rankInPset();

  std::string descriptor = getStreamDescriptorBetweenIONandCN(ionStreamType, psetNumber, psetNumber, rankInPset, nrPsets, psetSize, channel);

  return createStream(descriptor, false);
}

#if defined REROUTE_FFTW2_MALLOC
void *my_fftw_malloc(size_t n) {
  // don't use malloc() as it throws a bad_alloc on the BGP CNK.
  return new char[n];
}

void my_fftw_free(void *p) {
  delete[] static_cast<char*>(p);
}
#endif

int main(int argc, char **argv)
{
  std::clog.rdbuf(std::cout.rdbuf());
  
#if defined REROUTE_FFTW2_MALLOC
  fftw_malloc_hook = my_fftw_malloc;
  fftw_free_hook   = my_fftw_free;
#endif

  try {

#if defined HAVE_MPI
    MPI_Init(&argc, &argv);
#else
    (void) argc;
    (void) argv;
#endif

    LocationInfo locationInfo;

#if defined HAVE_LOG4CPLUS
    INIT_LOGGER( "CNProc" );
#elif defined HAVE_LOG4CXX
    #error LOG4CXX support is broken (nonsensical?) -- please fix this code if you want to use it
    Context::initialize();
    setLevel("Global",8);
#else
    INIT_LOGGER_WITH_SYSINFO(str(boost::format("CNProc@%04d") % locationInfo.rank()));
#endif

    if (locationInfo.rank() == 0) {
      locationInfo.print();

#if !defined HAVE_PKVERSION    
      std::string type = "brief";
      Version::show<CNProcVersion> (std::cout, "CNProc", type);
#endif
    }

    LOG_INFO_STR("Core " << locationInfo.rank() << " is core " << locationInfo.rankInPset() << " in pset " << locationInfo.psetNumber());

    getIONstreamType();

    if (LOG_CONDITION)
      LOG_DEBUG("Creating connection to ION ...");

    std::vector<SmartPtr<Stream> > ionStreams;

#if defined CLUSTER_SCHEDULING
    ionStreams.resize(locationInfo.nrPsets());

    for (unsigned ionode = 0; ionode < locationInfo.nrPsets(); ionode ++) {
      std::string descriptor = getStreamDescriptorBetweenIONandCN(ionStreamType, ionode, locationInfo.psetNumber(), locationInfo.rankInPset(), locationInfo.nrPsets(), locationInfo.psetSize(), 0);
      ionStreams[ionode] = createStream(descriptor, false);
    }

    Stream *controlStream = ionStreams[locationInfo.psetNumber()].get();
#else
    ionStreams.resize(1);
    ionStreams[0] = createIONstream(0, locationInfo);

    Stream *controlStream = ionStreams[0].get();
#endif

    if (LOG_CONDITION)
      LOG_DEBUG("Creating connection to ION: done");


    // an allocator for our big memory structures
#if defined HAVE_BGP    
    // The BG/P compute nodes have a flat memory space (no virtual memory), so memory can fragment, preventing us
    // from allocating big blocks. We thus put the big blocks in a separate arena.
    MallocedArena                bigArena(400*1024*1024, 32);
    SparseSetAllocator           bigAllocator(bigArena);
#else
    // assume memory is freely available
    HeapAllocator                bigAllocator;
#endif    

    SmartPtr<Parset>		 parset;
    SmartPtr<CN_Processing_Base> proc;
    CN_Command			 command;

    do {
      command.read(controlStream);
      //LOG_DEBUG_STR("Received command " << command.value() << " = " << command.name());

      switch (command.value()) {
	case CN_Command::PREPROCESS :	try {
                                          unsigned firstBlock = command.param();

					  parset = new Parset(controlStream);

				          switch (parset->nrBitsPerSample()) {
                                            case 4:  proc = new CN_Processing<i4complex>(*parset, ionStreams, &createIONstream, locationInfo, bigAllocator, firstBlock);
                                                     break;

                                            case 8:  proc = new CN_Processing<i8complex>(*parset, ionStreams, &createIONstream, locationInfo, bigAllocator, firstBlock);
                                                     break;

                                            case 16: proc = new CN_Processing<i16complex>(*parset, ionStreams, &createIONstream, locationInfo, bigAllocator, firstBlock);
                                                     break;
                                          }
                                        } catch (Exception &ex) {
                                          LOG_ERROR_STR("Caught Exception: " << ex);
                                        } catch (std::exception &ex) {
                                          LOG_ERROR_STR("Caught Exception: " << ex.what());
                                        } catch (...) {
                                          LOG_ERROR_STR("Caught Exception: unknown");
                                        }

#if 0 // FIXME: leads to deadlock when using TCP
					{
					  char failed = proc == 0;
					  ionStream->write(&failed, sizeof failed);
					}
#endif

					break;

	case CN_Command::PROCESS :	proc->process(command.param());
					break;

	case CN_Command::POSTPROCESS :	// proc == 0 if PREPROCESS threw an exception, after which all cores receive a POSTPROCESS message
					delete proc.release();
					delete parset.release();

#if defined HAVE_BGP // only SparseAllocator keeps track of its allocations
                                        if (!bigAllocator.empty())
                                          LOG_ERROR("Memory leak detected in bigAllocator");
#endif
					break;

	case CN_Command::STOP :		break;

	default :			LOG_FATAL("Bad command!");
					abort();
      }
    } while (command.value() != CN_Command::STOP);

#if defined HAVE_MPI
    MPI_Finalize();
    usleep(500 * locationInfo.rank()); // do not dump stats all at the same time
#endif
    
    return 0;
  } catch (Exception &ex) {
    LOG_FATAL_STR("Uncaught Exception: " << ex);
    return 1;
  }
}
