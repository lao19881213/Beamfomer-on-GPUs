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
//#  $Id: ION_main.cc 15296 2010-03-24 10:19:41Z romein $

#include <lofar_config.h>

#include <CommandServer.h>
#include <Common/Exceptions.h>
#include <Common/LofarLogger.h>
#include <Common/SystemCallException.h>
#include <GlobalVars.h>
#include <Job.h>
#include <JobQueue.h>
#include <Scheduling.h>
#include <Stream/SocketStream.h>

#include <string>
#include <boost/lexical_cast.hpp>

//#if defined HAVE_MPI
//#include <mpi.h>
//#endif


namespace LOFAR {
namespace RTCP {


void CommandServer::handleCommand(const std::string &command)
{
  //LOG_DEBUG_STR("command \"" << command << "\" received");

  if (command.compare(0, 7, "cancel ") == 0) {
    if (myPsetNumber == 0) {
      if (command.compare(0, 10, "cancel all") == 0)
	jobQueue.cancelAll();
      else
	jobQueue.cancel(boost::lexical_cast<unsigned>(command.substr(7)));
    }
  } else if (command == "list_jobs") {
    if (myPsetNumber == 0)
      jobQueue.listJobs();
  } else if (command.compare(0, 7, "parset ") == 0) {
    jobQueue.insert(new Job(command.substr(7).c_str()));
    itsNrJobsCreated.up();
  } else if (command == "quit") {
    itsQuit = true;
  } else if (command == "threads") {
    ThreadMap::instance().report();
#if defined HAVE_BGP    
  } else if (command == "debug") {
    LOGCOUT_SETLEVEL(8);
  } else if (command == "nodebug") {
    LOGCOUT_SETLEVEL(4);
#endif    
  } else if (command == "") {
    // quietly drop empty commands
  } else if (myPsetNumber == 0) {
    LOG_ERROR_STR("command \"" << command << "\" not understood");
  }
}


void CommandServer::commandMaster()
{
#if defined HAVE_BGP_ION
  //doNotRunOnCore0();
  runOnCore0();
#endif

  std::vector<SmartPtr<MultiplexedStream> > ionStreams(nrPsets);

  for (unsigned ion = 1; ion < nrPsets; ion ++)
    ionStreams[ion] = new MultiplexedStream(*allIONstreamMultiplexers[ion], 0);

#if defined HAVE_BGP
  SocketStream sk("0.0.0.0", 4000, SocketStream::TCP, SocketStream::Server);
#else
  SocketStream sk("0.0.0.0", 3999, SocketStream::TCP, SocketStream::Server);
#endif

  LOG_INFO("Command server ready");

  while (!itsQuit) {
    std::string command;

    try {
      command = sk.readLine();
      LOG_INFO_STR("read command: " << command);
    } catch (Stream::EndOfStreamException &) {
      sk.reaccept();
      continue;
    }

    unsigned size = command.size() + 1;

    //MPI_Bcast(&size, sizeof size, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(const_cast<char *>(command.c_str()), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    for (unsigned ion = 1; ion < nrPsets; ion ++) {
      ionStreams[ion]->write(&size, sizeof size);
      ionStreams[ion]->write(command.c_str(), size);
    }

    try {
      handleCommand(command);
    } catch (Exception &ex) {
      LOG_ERROR_STR("handleCommand caught Exception: " << ex);
    } catch (std::exception &ex) {
      LOG_ERROR_STR("handleCommand caught std::exception: " << ex.what());
    } catch (...) {
      LOG_ERROR("handleCommand caught non-std::exception: ");
      throw;
    }
  }
}


void CommandServer::commandSlave()
{
  MultiplexedStream streamFromMaster(*allIONstreamMultiplexers[0], 0);

  while (!itsQuit) {
    unsigned size;

    //MPI_Bcast(&size, sizeof size, MPI_INT, 0, MPI_COMM_WORLD);
    streamFromMaster.read(&size, sizeof size);

    std::vector<char> command(size);
    //MPI_Bcast(command, size, MPI_CHAR, 0, MPI_COMM_WORLD);
    streamFromMaster.read(&command[0], size);

    try {
      handleCommand(&command[0]);
    } catch (Exception &ex) {
      LOG_ERROR_STR("handleCommand caught Exception: " << ex);
    } catch (std::exception &ex) {
      LOG_ERROR_STR("handleCommand caught std::exception: " << ex.what());
    } catch (...) {
      LOG_ERROR("handleCommand caught non-std::exception: ");
      throw;
    }
  }
}


void CommandServer::jobCleanUpThread()
{
  while (itsNrJobsCreated.down()) {
    Job *job = finishedJobs.remove();
    jobQueue.remove(job);
    delete job;
  }
}


CommandServer::CommandServer()
:
  itsQuit(false)
{
}


void CommandServer::start()
{
  itsJobCleanUpThread = new Thread(this, &CommandServer::jobCleanUpThread, "JobCleanUpThread", 65536);

  if (myPsetNumber == 0)
    commandMaster();
  else
    commandSlave();
}


CommandServer::~CommandServer()
{
  itsNrJobsCreated.noMore();
}


} // namespace RTCP
} // namespace LOFAR
