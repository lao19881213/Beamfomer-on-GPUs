#include "lofar_config.h"
#include <StorageProcesses.h>
#include <sys/time.h>
#include <unistd.h>
#include <Common/Thread/Thread.h>
#include <Stream/PortBroker.h>
#include <SSH.h>
#include <boost/format.hpp>

namespace LOFAR {
namespace RTCP {

using namespace std;
using boost::format;

/*
 * Manage a Storage_main process (RTCP/Storage). The control sequence is as follows:
 *
 * hostList = "OLAP.Storage.hosts"
 *
 * 1. Create a StorageProcesses manager object:
 *      manager = StorageProcesses(parset, logprefix)
 * 2. Spawn all Storage processes:
 *      manager.start()
 *    Which will ssh to all hosts in hostList and start Storage_main,
 *    and establish a control channel to each.
 * 3. Let your application connect to whatever is indicated by
 *      getStreamDescriptorBetweenIONandStorage(parset, outputType, streamNr)
 * 4. After the observation, generate and forward the final metadata by calling
 *      manager.forwardFinalMetaData( deadline )
 * 5. Optionally, wait for the Storage_main processes to finish with a given
 * deadline:
 *      manager.stop( deadline )
 */


StorageProcess::StorageProcess( StorageProcesses &manager, const Parset &parset, const string &logPrefix, int rank, const string &hostname )
:
  itsManager(manager),
  itsParset(parset),
  itsLogPrefix(str(boost::format("%s [StorageWriter rank %2d host %s] ") % logPrefix % rank % hostname)),
  itsRank(rank),
  itsHostname(hostname)
{
}


StorageProcess::~StorageProcess()
{
  // cancel the control thread in case it is still active
  itsThread->cancel();
}


void StorageProcess::start()
{
  std::string userName   = itsParset.getString("OLAP.Storage.userName");
  std::string sshKey     = itsParset.getString("OLAP.Storage.sshIdentityFile");
  std::string executable = itsParset.getString("OLAP.Storage.msWriter");

  char cwd[1024];

  if (getcwd(cwd, sizeof cwd) == 0)
    THROW_SYSCALL("getcwd");

  std::string commandLine = str(boost::format("cd %s && %s%s %u %d %u 2>&1")
    % cwd
#if defined USE_VALGRIND
    % "valgrind --leak-check=full "
#else
    % ""
#endif
    % executable
    % itsParset.observationID()
    % itsRank
#if defined WORDS_BIGENDIAN
    % 1
#else
    % 0
#endif
  );

  itsSSHconnection = new SSHconnection(itsLogPrefix, itsHostname, commandLine, userName, sshKey, 0);
  itsSSHconnection->start();

  itsThread = new Thread(this, &StorageProcess::controlThread, itsLogPrefix + "[ControlThread] ", 65535);
}


void StorageProcess::stop(struct timespec deadline)
{
  if (!itsSSHconnection) {
    // never started
    return;
  }

  itsSSHconnection->wait(deadline);

  itsThread->cancel();
}


bool StorageProcess::isDone()
{
  return itsSSHconnection->isDone();
}


void StorageProcess::controlThread()
{
  // Connect
  LOG_DEBUG_STR(itsLogPrefix << "[ControlThread] connecting...");
  std::string resource = getStorageControlDescription(itsParset.observationID(), itsRank);
  PortBroker::ClientStream stream(itsHostname, storageBrokerPort(itsParset.observationID()), resource, 0);

  // Send parset
  LOG_DEBUG_STR(itsLogPrefix << "[ControlThread] connected -- sending parset");
  itsParset.write(&stream);
  LOG_DEBUG_STR(itsLogPrefix << "[ControlThread] sent parset");

  // Send final meta data once it is available
  itsManager.itsFinalMetaDataAvailable.down();

  LOG_DEBUG_STR(itsLogPrefix << "[ControlThread] sending final meta data");
  itsManager.itsFinalMetaData.write(stream);
  LOG_DEBUG_STR(itsLogPrefix << "[ControlThread] sent final meta data");
}


StorageProcesses::StorageProcesses( const Parset &parset, const std::string &logPrefix )
:
  itsParset(parset),
  itsLogPrefix(logPrefix)
{
}

StorageProcesses::~StorageProcesses()
{
  // never let any processes linger
  stop(0);
}


void StorageProcesses::start()
{
  vector<string> hostnames = itsParset.getStringVector("OLAP.Storage.hosts");

  itsStorageProcesses.resize(hostnames.size());

  LOG_DEBUG_STR(itsLogPrefix << "Starting " << itsStorageProcesses.size() << " Storage processes");

  for (unsigned rank = 0; rank < itsStorageProcesses.size(); rank ++) {
    itsStorageProcesses[rank] = new StorageProcess(*this, itsParset, itsLogPrefix, rank, hostnames[rank]);
    itsStorageProcesses[rank]->start();
  }  
}


void StorageProcesses::stop( time_t deadline )
{
  LOG_DEBUG_STR(itsLogPrefix << "Stopping storage processes");

  size_t nrRunning = 0;

  for (unsigned rank = 0; rank < itsStorageProcesses.size(); rank ++)
    if (itsStorageProcesses[rank].get())
      nrRunning++;

  while(nrRunning > 0) {
    for (unsigned rank = 0; rank < itsStorageProcesses.size(); rank ++) {
      if (!itsStorageProcesses[rank].get())
        continue;

      if (itsStorageProcesses[rank]->isDone() || time(0) >= deadline) {
        struct timespec immediately = { 0, 0 };

        itsStorageProcesses[rank]->stop(immediately);
        itsStorageProcesses[rank] = 0;

        nrRunning--;
      }
    }  

    if (nrRunning > 0)
      sleep(1);
  }

  itsStorageProcesses.clear();

  LOG_DEBUG_STR(itsLogPrefix << "Storage processes are stopped");
}


void StorageProcesses::forwardFinalMetaData( time_t deadline )
{
  struct timespec deadline_ts = { deadline, 0 };

  Thread thread(this, &StorageProcesses::finalMetaDataThread, itsLogPrefix + "[FinalMetaDataThread] ", 65536);

  // abort the thread if deadline passes
  try {
    if (!thread.wait(deadline_ts)) {
      LOG_WARN_STR(itsLogPrefix << "Cancelling FinalMetaDataThread");

      thread.cancel();
    }
  } catch(...) {
    thread.cancel();
    throw;
  }
}


void StorageProcesses::finalMetaDataThread()
{
  std::string hostName    = itsParset.getString("OLAP.FinalMetaDataGatherer.host");
  std::string userName    = itsParset.getString("OLAP.FinalMetaDataGatherer.userName");
  std::string sshKey      = itsParset.getString("OLAP.FinalMetaDataGatherer.sshIdentityFile");
  std::string executable  = itsParset.getString("OLAP.FinalMetaDataGatherer.executable");

  char cwd[1024];

  if (getcwd(cwd, sizeof cwd) == 0)
    THROW_SYSCALL("getcwd");

  std::string commandLine = str(boost::format("cd %s && %s %d 2>&1")
      % cwd
      % executable
      % itsParset.observationID()
      );

  // Start the remote process
  SSHconnection sshconn(itsLogPrefix + "[FinalMetaData] ", hostName, commandLine, userName, sshKey);
  sshconn.start();

  // Connect
  LOG_DEBUG_STR(itsLogPrefix << "[FinalMetaData] [ControlThread] connecting...");
  std::string resource = getStorageControlDescription(itsParset.observationID(), -1);
  PortBroker::ClientStream stream(hostName, storageBrokerPort(itsParset.observationID()), resource, 0);

  // Send parset
  LOG_DEBUG_STR(itsLogPrefix << "[FinalMetaData] [ControlThread] connected -- sending parset");
  itsParset.write(&stream);
  LOG_DEBUG_STR(itsLogPrefix << "[FinalMetaData] [ControlThread] sent parset");

  // Receive final meta data
  itsFinalMetaData.read(stream);
  LOG_DEBUG_STR(itsLogPrefix << "[FinalMetaData] [ControlThread] obtained final meta data");

  // Notify clients
  itsFinalMetaDataAvailable.up(itsStorageProcesses.size());

  // Wait for or end the remote process
  sshconn.wait();
}

}
}
