//# TBB_Writer_main.cc: LOFAR Transient Buffer Boards (TBB) Data Writer
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands.
//#
//# This program is free software: you can redistribute it and/or modify
//# it under the terms of the GNU General Public License as published by
//# the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite.  If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: TBB_Writer_main.cc 27385 2013-11-13 15:34:52Z amesfoort $

#include <lofar_config.h>               // before any other include

#define _FILE_OFFSET_BITS 64
#include <cstddef>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <cerrno>
#include <libgen.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>

#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include <Common/LofarLogger.h>
#include <Common/StringUtil.h>
#include <Common/NewHandler.h>
#include <ApplCommon/StationConfig.h>
#include <ApplCommon/AntField.h>
#include <CoInterface/Exceptions.h>

#ifdef HAVE_DAL
#  include <dal/lofar/StationNames.h>
#else // some builds only need BF MS output, so do not block the build without DAL
#  warning The TBB writer may be built without DAL, but will not write any output
#endif

#include "TBB_Writer.h"
#include "IOPriority.h"

#define TBB_DEFAULT_BASE_PORT   0x7bb0  // i.e. tbb0
#define TBB_DEFAULT_LAST_PORT   0x7bbb  // 0x7bbf for NL, 0x7bbb for int'l stations

#define STDLOG_BUFFER_SIZE      1024

using namespace std;

struct progArgs {
  string parsetFilename;
  string stCalTablesDir;
  string antFieldDir;
  string outputDir;
  string input;
  uint16_t port;
  struct timeval timeoutVal;
  bool keepRunning;
};

static char stdoutbuf[STDLOG_BUFFER_SIZE];
static char stderrbuf[STDLOG_BUFFER_SIZE];

LOFAR::NewHandler badAllocExcHandler(LOFAR::BadAllocException::newHandler);

static volatile bool sigint_seen;

static void termSigsHandler(int sig_nr)
{
  if (sig_nr == SIGINT) {
    /*
     * For graceful user abort. Signal might be missed, but timeout
     * catches it later, so don't bother with cascaded signals.
     */
    sigint_seen = true;
  }
}

/*
 * Register signal handlers for SIGINT and SIGTERM to gracefully terminate early,
 * so we can break out of blocking system calls and exit without corruption of already written output.
 * Leave SIGQUIT (Ctrl-\) untouched, so users can still easily quit immediately.
 */
static void setTermSigsHandler()
{
  struct sigaction sa;

  sa.sa_handler = termSigsHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  int err = sigaction(SIGINT,  &sa, NULL); // keyb INT (typically Ctrl-C)
  err |= sigaction(SIGTERM, &sa, NULL);
  err |= sigaction(SIGALRM, &sa, NULL);    // for setitimer(); don't use sleep(3) and friends
  if (err != 0) {
    LOG_WARN("TBB: Failed to register SIGINT/SIGTERM handler to allow manual, early, graceful program termination.");
  }
}

static vector<string> getTBB_InputStreamNames(const string& input, uint16_t portsBase)
{
  int nTbbBoards;
  try {
    LOFAR::StationConfig stConf;
    nTbbBoards = stConf.nrTBBs;
  } catch (LOFAR::AssertError& ) { // config file not found
    LOG_DEBUG_STR("Falling back to at most " << TBB_DEFAULT_LAST_PORT - TBB_DEFAULT_BASE_PORT + 1 << " input streams (1 per board)");
    nTbbBoards = TBB_DEFAULT_LAST_PORT - TBB_DEFAULT_BASE_PORT + 1; // fallback
  }

  vector<string> allInputStreamNames;
  if (input == "udp" || input == "tcp") {
    for (uint16_t port = portsBase; port <= portsBase + nTbbBoards; ++port) {
      // 0.0.0.0: could restrict to station IPs/network, but need netmask lookup and allow localhost. Not critical: data arrives on a separate VLAN.
      string streamName(input + ":0.0.0.0:" + LOFAR::formatString("%hu", port));
      allInputStreamNames.push_back(streamName);
    }
  } else { // file or named pipe input
    size_t colonPos = input.find(':');
    if (colonPos == string::npos) {
      return allInputStreamNames;
    }
    size_t placeholderPos = input.find_last_of('%');
    if (placeholderPos == string::npos) { // single input, no expansion needed
      if (access(input.c_str() + colonPos + 1, R_OK) == 0) {
        allInputStreamNames.push_back(input);
      }
    } else { // expand e.g. file:x%y-%.raw into {file:x%y-0.raw, file:x%y-1.raw, ..., file:x%y-11.raw}
      for (int i = 0; i < nTbbBoards; ++i) {
        string streamName(input);
        streamName.replace(placeholderPos, 1, LOFAR::formatString("%u", i));
        if (access(streamName.c_str() + colonPos + 1, R_OK) == 0) {
          allInputStreamNames.push_back(streamName);
        }
      }
    }
  }

  return allInputStreamNames;
}

static void retrieveStationCalTables(string& stCalTablesDir)
{
  /*
   * Users need the station calibration tables included. This is a major pain, because
   * we figure out which station(s) we receive from at runtime (relying on the static
   * mapping is a disaster waiting to happen), we cannot ask the stations and the
   * alternative, from svn, is unreliable and races with (few) Science Support updates.
   * Not all users care about the race, a few do. Also, auth, and this exposes an internal
   * interface (cal tables) to users... Still do it: TBB is too low prio to get stuff nice.
   *
   * Get tables from all stations for the right cal mode (i.e. usually only verifies svn local copy),
   * Run 'svn cleanup' and 'svn upgrade' when needed, otherwise remove the local copies and re-retrieve.
   *

   */

  //svn checkout https://svn.astron.nl/Station/trunk/CalTables
  //but only the needed files
  //svn update
  //Ctrl-C doesn't seem to kill svn co/up (only pause/halt), so use Ctrl-\ (QUIT), then svn cleanup

  //svn: Working copy '.' locked
  //svn: run 'svn cleanup' to remove locks (type 'svn help cleanup' for details)
  //svn cleanup

  //rm -rf CalTables

  // Note: include the entire cal table as-is, because that easily allows users to just resort to the raw files

  //	- if stCalTablesDir.empty():

  //	- get station names, st cal mode
  //	- fork process (sh script), do data writes
  //	- sh script does svn checkout/update on req files only into ~/TBB_Writer-Station-CalTabs-localcopy/Station/CalTables/*
  //	- listen for tbb data. When data writes done, do timed wait() on script pid, and if ok, add cal tables.
  //	- if not ok: if timeout { signal script to abort and run svn cleanup, wait()}. Skip writing cal tabs, log warning + script output.

}

static int antSetName2AntFieldIndex(const string& antSetName)
{
  int idx;

  if (strncmp(antSetName.c_str(), "LBA", sizeof("LBA") - 1) == 0) {
    idx = LOFAR::AntField::LBA_IDX;
  } else if (strncmp(antSetName.c_str(), "HBA_ZERO", sizeof("HBA_ZERO") - 1) == 0) {
    idx = LOFAR::AntField::HBA0_IDX;
  } else if (strncmp(antSetName.c_str(), "HBA_ONE", sizeof("HBA_ONE") - 1) == 0) {
    idx = LOFAR::AntField::HBA1_IDX;
  } else if (strncmp(antSetName.c_str(), "HBA", sizeof("HBA") - 1) == 0) {
    idx = LOFAR::AntField::HBA_IDX;
  } else {
    throw LOFAR::Cobalt::StorageException("unknown antenna set name");
  }

  return idx;
}

static LOFAR::Cobalt::StationMetaDataMap getExternalStationMetaData(const LOFAR::Cobalt::Parset& parset, const string& antFieldDir)
{
  LOFAR::Cobalt::StationMetaDataMap stMdMap;

  try {
    // Find path to antenna field files. If not a prog arg, try via $LOFARROOT, else via parset.
    // LOFAR repos location: MAC/Deployment/data/StaticMetaData/AntennaFields/
    string antFieldPath(antFieldDir);
    if (antFieldPath.empty()) {
      char* lrpath = getenv("LOFARROOT");
      if (lrpath != NULL) {
        antFieldPath = string(lrpath) + "/etc/StaticMetaData/";
      } else { // parset typically gives "/data/home/lofarsys/production/lofar/etc/StaticMetaData"
        antFieldPath = parset.AntennaFieldsDir(); // doesn't quite do what its name suggests, so append a component
        if (!antFieldPath.empty()) {
          antFieldPath.push_back('/');
        }
      }
      antFieldPath.append("AntennaFields/");
    }

    int fieldIdx = antSetName2AntFieldIndex(parset.antennaSet());

    vector<string> stationNames(parset.allStationNames());
    for (vector<string>::const_iterator it(stationNames.begin());
         it != stationNames.end(); ++it) {

      string stName(it->substr(0, sizeof("CS001") - 1)); // drop any "HBA0"-like suffix
      string antFieldFilename(antFieldPath + stName + "-AntennaField.conf");

      // Tries to locate the filename if no abs path is given, else throws AssertError exc.
      LOFAR::AntField antField(antFieldFilename);

      // Compute absolute antenna positions from centre + relative.
      // See AntField.h in ApplCommon for the AFArray typedef and contents (first is shape, second is values).
      LOFAR::Cobalt::StationMetaData stMetaData;
      stMetaData.available = true;
      stMetaData.antPositions = antField.AntPos(fieldIdx).second;
      for (size_t i = 0; i < stMetaData.antPositions.size(); i += 3) {
        stMetaData.antPositions.at(i + 2) += antField.Centre(fieldIdx).second.at(2);
        stMetaData.antPositions[i + 1] += antField.Centre(fieldIdx).second[1];
        stMetaData.antPositions[i] += antField.Centre(fieldIdx).second[0];
      }

      stMetaData.normalVector = antField.normVector(fieldIdx).second;
      stMetaData.rotationMatrix = antField.rotationMatrix(fieldIdx).second;

#ifdef HAVE_DAL
      stMdMap.insert(make_pair(dal::stationNameToID(stName), stMetaData));
#endif
    }
  } catch (LOFAR::AssertError& exc) {
    // Throwing AssertError already sends a message to the logger.
#ifdef HAVE_DAL
  } catch (dal::DALValueError& exc) {
    throw LOFAR::Cobalt::StorageException(exc.what());
#endif
  }

  return stMdMap;
}

static int doTBB_Run(const vector<string>& inputStreamNames, const LOFAR::Cobalt::Parset& parset,
                     const LOFAR::Cobalt::StationMetaDataMap& stMdMap, struct progArgs& args)
{
  string logPrefix("TBB obs " + LOFAR::formatString("%u", parset.observationID()) + ": ");

  vector<int> thrExitStatus(2 * inputStreamNames.size(), 0);
  int err = 1;
  try {
#ifdef HAVE_DAL
    // When this obj goes out of scope, worker threads are cancelled and joined with.
    LOFAR::Cobalt::TBB_Writer writer(inputStreamNames, parset, stMdMap, args.outputDir, logPrefix, thrExitStatus);
#else
    // Allow building without DAL (some users don't need TBB_Writer), but bail if run.
    throw LOFAR::APSException("TBB_Writer needs but was not built with DAL");
#endif

    /*
     * We don't know how much data comes in, so cancel workers when all are idle for a while (timeoutVal).
     * In some situations, threads can become active again after idling a bit, so periodically monitor thread timeout stamps.
     * Poor man's sync, but per-thread timers to break read() to notify us of idleness does not work.
     * This (sucks and :)) could be improved once the LOFAR system tells us how much data will be dumped, or when done.
     */
    struct itimerval timer = {args.timeoutVal, args.timeoutVal};
    if (setitimer(ITIMER_REAL, &timer, NULL) != 0) {
      THROW_SYSCALL("setitimer");
    }

    bool anyFrameReceived = false; // don't quit if there is no data immediately after starting
    size_t nrWorkersDone;
    do {
      pause();
      if (sigint_seen) { // typically Ctrl-C
        args.keepRunning = false; // for main(), not for worker threads
        break;
      }

      nrWorkersDone = 0;
      for (size_t i = 0; i < inputStreamNames.size(); i++) {
        struct timeval now;
        gettimeofday(&now, NULL);
        time_t lastActive_sec = writer.getTimeoutStampSec(i);
        if (lastActive_sec != 0) {
          anyFrameReceived = true;
        }
        if (anyFrameReceived && lastActive_sec <= now.tv_sec - args.timeoutVal.tv_sec) {
          nrWorkersDone += 1;
        }
      }
    } while (nrWorkersDone < inputStreamNames.size());
    err = 0;
  } catch (LOFAR::Exception& exc) {
    LOG_FATAL_STR(logPrefix << "LOFAR::Exception: " << exc);
  } catch (exception& exc) {
    LOG_FATAL_STR(logPrefix << "std::exception: " << exc.what());
  }

  // Propagate exit status != 0 from any input or output worker thread.
  for (unsigned i = 0; i < thrExitStatus.size(); ++i) {
    if (thrExitStatus[i] != 0) {
      err = 1;
      break;
    }
  }

  return err;
}

static int isExistingDirname(const string& dirname)
{
  struct stat st;

  if (stat(dirname.c_str(), &st) != 0) {
    return errno;
  }

  // Check if the last component is a dir too (stat() did the rest).
  if (!S_ISDIR(st.st_mode)) {
    return ENOTDIR;
  }

  return 0;
}

static void printUsage(const char* progname)
{
  cout << "LOFAR TBB_Writer version: ";
#ifndef TBB_WRITER_VERSION
  cout << LOFAR::StorageVersion::getVersion();
#else
  cout << TBB_WRITER_VERSION;
#endif
  cout << endl;
  cout << "Write incoming LOFAR TBB data with meta data to storage in HDF5 format." << endl;
  cout << "Usage: " << progname << " -p parsets/L12345.parset [OPTION]..." << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  -p, --parset=L12345.parset          path to file with observation settings (mandatory)" << endl;
  cout << endl;
  cout << "  -c, --stcaltablesdir=/c/CalTables   path to override SVN retrieval of station calibration tables (like CS001/CalTable_001_mode1.dat)" << endl;
  cout << "  -a, --antfielddir=/a/AntennaFields  path to override $LOFARROOT and parset path for antenna field files (like CS001-AntennaField.conf)" << endl;
  cout << "  -o, --outputdir=tbbout              existing output directory" << endl;
  cout << "  -i, --input=tcp|udp                 input stream(s) or type (default: udp)" << endl;
  cout << "              file:raw.dat                if file or pipe name has a '%'," << endl;
  cout << "              pipe:named-%.pipe           then the last '%' is replaced by 0, 1, ..., 11" << endl;
  cout << "  -b, --portbase=31665                start of range of 12 consecutive udp/tcp ports to receive from" << endl;
  cout << "  -t, --timeout=10                    seconds of input inactivity until dump is considered completed" << endl;
  cout << endl;
  cout << "  -k, --keeprunning[=true|false]      accept new input after a dump completed (default: true)" << endl;
  cout << endl;
  cout << "  -h, --help                          print program name, version number and this info, then exit" << endl;
  cout << "  -v, --version                       same as --help" << endl;
}

static int parseArgs(int argc, char *argv[], struct progArgs* args)
{
  int status = 0;

  // Default values
  args->parsetFilename = "";    // there is no default parset filename, so not passing it is fatal
  args->stCalTablesDir = "";    // idem, but otherwise, retrieve from svn and not fatal
  args->antFieldDir = "";       // idem, but otherwise, detect and not fatal

  args->outputDir = "";
  args->input = "udp";
  args->port = TBB_DEFAULT_BASE_PORT;
  args->timeoutVal.tv_sec = 10; // after this default of inactivity cancel all input threads and close output files
  args->timeoutVal.tv_usec = 0;
  args->keepRunning = true;

  static const struct option long_opts[] = {
    // NOTE: If you change this, then also change the code below AND the printUsage() code above!
    // {const char *name, int has_arg, int *flag, int val}
    {"parset",         required_argument, NULL, 'p'},
    {"stcaltablesdir", required_argument, NULL, 'c'}, // station calibration tables
    {"antfielddir",    required_argument, NULL, 'a'}, // antenna field info
    {"outputdir",      required_argument, NULL, 'o'},
    {"input",          required_argument, NULL, 'i'},
    {"portbase",       required_argument, NULL, 'b'}, // port (b)ase
    {"timeout",        required_argument, NULL, 't'},

    {"keeprunning",    optional_argument, NULL, 'k'},

    {"help",           no_argument,       NULL, 'h'},
    {"version",        no_argument,       NULL, 'v'},

    {NULL, 0, NULL, 0}
  };

  opterr = 0; // prevent error printing to stderr by getopt_long()
  int opt, err;
  while ((opt = getopt_long(argc, argv, "hvs:a:o:p:b:t:k::", long_opts, NULL)) != -1) {
    switch (opt) {
    case 'p':
      args->parsetFilename = optarg;
      break;
    case 'c':
      args->stCalTablesDir = optarg;
      if (args->stCalTablesDir[0] != '\0' && args->stCalTablesDir[args->stCalTablesDir.size() - 1] != '/') {
        args->stCalTablesDir.push_back('/');
      }
      if ((err = isExistingDirname(args->stCalTablesDir)) != 0) {
        LOG_FATAL_STR("TBB: station cal tab dir argument value " << optarg << ": " << strerror(err));
        status = 1;
      }
      break;
    case 'a':
      args->antFieldDir = optarg;
      if (args->antFieldDir[0] != '\0' && args->antFieldDir[args->antFieldDir.size() - 1] != '/') {
        args->antFieldDir.push_back('/');
      }
      if ((err = isExistingDirname(args->antFieldDir)) != 0) {
        LOG_FATAL_STR("TBB: antenna field dir argument value " << optarg << ": " << strerror(err));
        status = 1;
      }
      break;
    case 'o':
      args->outputDir = optarg;
      if (args->outputDir[0] != '\0' && args->outputDir[args->outputDir.size() - 1] != '/') {
        args->outputDir.push_back('/');
      }
      if ((err = isExistingDirname(args->outputDir)) != 0) {
        LOG_FATAL_STR("TBB: output dir argument value " << optarg << ": " << strerror(err));
        status = 1;
      }
      break;
    case 'i':
      if (strcmp(optarg, "tcp") == 0 || strcmp(optarg, "udp") == 0 ||
          strncmp(optarg, "file:", sizeof("file:") - 1) == 0 ||
          strncmp(optarg, "pipe:", sizeof("pipe:") - 1) == 0) {
        args->input = optarg;
      } else {
        LOG_FATAL_STR("TBB: Invalid input argument value: " << optarg);
        status = 1;
      }
      break;
    case 'b':
      try {
        args->port = boost::lexical_cast<uint16_t>(optarg);
        if (args->port > 65536 - (TBB_DEFAULT_LAST_PORT - TBB_DEFAULT_BASE_PORT)) {
          throw boost::bad_lexical_cast(); // abuse exc type to have single catch
        }
      } catch (boost::bad_lexical_cast& /*exc*/) {
        LOG_FATAL_STR("TBB: Invalid port argument value: " << optarg);
        status = 1;
      }
      break;
    case 't':
      try {
        args->timeoutVal.tv_sec = boost::lexical_cast<unsigned long>(optarg);
      } catch (boost::bad_lexical_cast& /*exc*/) {
        LOG_FATAL_STR("TBB: Invalid timeout argument value: " << optarg);
        status = 1;
      }
      break;
    case 'k':
      if (optarg == NULL || optarg[0] == '\0') {
        args->keepRunning = true;
        break;
      }
      try {
        args->keepRunning = boost::lexical_cast<bool>(optarg);
      } catch (boost::bad_lexical_cast& /*exc*/) {
        LOG_FATAL_STR("TBB: Invalid keeprunning argument value: " << optarg);
        status = 1;
      }
      break;
    case 'h':
    case 'v':
      if (status == 0) {
        status = 2;
      }
      break;
    default: // '?'
      LOG_FATAL_STR("TBB: Invalid program argument or missing argument value: " << argv[optind - 1]);
      status = 1;
    }
  }

  if (optind < argc) {
    ostringstream oss;
    oss << "TBB: Failed to recognize arguments:";
    while (optind < argc) {
      oss << " " << argv[optind++]; // good enough
    }
    LOG_FATAL_STR(oss.str());
    status = 1;
  }

  return status;
}

int main(int argc, char* argv[])
{
  struct progArgs args;
  int err;

#if defined HAVE_LOG4CPLUS || defined HAVE_LOG4CXX
  struct Log {
    Log(const char* argv0)
    {
      char *dirc = strdup(argv0); // dirname() may clobber its arg
      if (dirc != NULL) {
        INIT_LOGGER(string(getenv("LOFARROOT") ? : dirname(dirc)) + "/../etc/outputProc.log_prop");
        free(dirc);
      }
    }

    ~Log()
    {
      LOGGER_EXIT_THREAD(); // destroys NDC created by INIT_LOGGER()
    }
  } logger(argv[0]);
#endif

  err = setvbuf(stdout, stdoutbuf, _IOLBF, sizeof stdoutbuf);
  err |= setvbuf(stderr, stderrbuf, _IOLBF, sizeof stderrbuf);
  if (err != 0) {
    LOG_WARN("TBB: failed to change stdout and/or stderr output buffers");
  }

  if ((err = parseArgs(argc, argv, &args)) != 0) {
    if (err == 2) err = 0;
    printUsage(argv[0]);
    return err;
  }

  setTermSigsHandler();

  const vector<string> inputStreamNames(getTBB_InputStreamNames(args.input, args.port));
  if (inputStreamNames.empty()) {
    LOG_FATAL("TBB: none of the input streams is accessible to read from");
    return 1;
  }

  retrieveStationCalTables(args.stCalTablesDir);

  // We don't run alone, so try to increase the QoS we get from the OS to decrease the chance of data loss.
  setIOpriority(); // reqs CAP_SYS_NICE or CAP_SYS_ADMIN
  setRTpriority(); // reqs CAP_SYS_NICE
  lockInMemory();  // reqs CAP_IPC_LOCK

  err = 1;
  try {
    LOFAR::Cobalt::Parset parset(args.parsetFilename);
    LOFAR::Cobalt::StationMetaDataMap stMdMap(getExternalStationMetaData(parset, args.antFieldDir));

    err = 0;
    do {
      err += doTBB_Run(inputStreamNames, parset, stMdMap, args);
    } while (args.keepRunning && err < 1000);
    if (err == 1000) { // Nr of dumps per obs was estimated to fit in 3 digits.
      LOG_FATAL("TBB: Reached max nr of errors seen. Shutting down to avoid filling up storage with logging crap.");
    }

    // Config exceptions (opening or parsing) are fatal. Too bad we cannot have it in one type.
  } catch (LOFAR::Cobalt::CoInterfaceException& exc) {
    LOG_FATAL_STR("TBB: Required parset key/values missing: " << exc);
  } catch (LOFAR::APSException& exc) {
    LOG_FATAL_STR("TBB: Parameterset error: " << exc);
  } catch (LOFAR::Cobalt::StorageException& exc) {
    LOG_FATAL_STR("TBB: Antenna field files: " << exc);
  }

  return err == 0 ? 0 : 1;
}

