//# OutputThread.cc:
//# Copyright (C) 2009-2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
//#
//# This file is part of the LOFAR software suite.
//# The LOFAR software suite is free software: you can redistribute it and/or
//# modify it under the terms of the GNU General Public License as published
//# by the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# The LOFAR software suite is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: OutputThread.cc 25646 2013-07-12 09:58:39Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include "OutputThread.h"

#include <cerrno>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include <boost/format.hpp>

#include <Common/StringUtil.h>
#include <Common/SystemCallException.h>
#include <Common/Thread/Mutex.h>
#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Cancellation.h>

#if defined HAVE_AIPSPP
#include <casa/Exceptions/Error.h>
#endif

#include "MSWriterFile.h"
#include "MSWriterCorrelated.h"
#include "MSWriterDAL.h"
#include "MSWriterNull.h"

namespace LOFAR
{
  namespace Cobalt
  {

    static Mutex makeDirMutex;
    static Mutex casacoreMutex;

    using namespace std;

    static void makeDir(const string &dirname, const string &logPrefix)
    {
      ScopedLock scopedLock(makeDirMutex);
      struct stat s;

      if (stat(dirname.c_str(), &s) == 0) {
        // path already exists
        if ((s.st_mode & S_IFMT) != S_IFDIR) {
          LOG_WARN_STR(logPrefix << "Not a directory: " << dirname);
        }
      } else if (errno == ENOENT) {
        // create directory
        LOG_DEBUG_STR(logPrefix << "Creating directory " << dirname);

        if (mkdir(dirname.c_str(), 0777) != 0 && errno != EEXIST) {
          THROW_SYSCALL(string("mkdir ") + dirname);
        }
      } else {
        // something else went wrong
        THROW_SYSCALL(string("stat ") + dirname);
      }
    }


    /* create a directory as well as all its parent directories */
    static void recursiveMakeDir(const string &dirname, const string &logPrefix)
    {
      using namespace boost;

      string curdir;
      vector<string> splitName;

      split(splitName, dirname, is_any_of("/"));

      for (unsigned i = 0; i < splitName.size(); i++) {
        curdir += splitName[i] + '/';
        makeDir(curdir, logPrefix);
      }
    }


    OutputThread::OutputThread(const Parset &parset, OutputType outputType, unsigned streamNr, Queue<SmartPtr<StreamableData> > &freeQueue, Queue<SmartPtr<StreamableData> > &receiveQueue, const std::string &logPrefix, bool isBigEndian, const std::string &targetDirectory)
      :
      itsParset(parset),
      itsOutputType(outputType),
      itsStreamNr(streamNr),
      itsIsBigEndian(isBigEndian),
      itsLogPrefix(logPrefix + "[OutputThread] "),
      itsTargetDirectory(targetDirectory),
      itsFreeQueue(freeQueue),
      itsReceiveQueue(receiveQueue),
      itsBlocksWritten(0),
      itsBlocksDropped(0),
      itsNrExpectedBlocks(0),
      itsNextSequenceNumber(0)
    {
    }


    void OutputThread::start()
    {
      itsThread = new Thread(this, &OutputThread::mainLoop, itsLogPrefix);
    }


    void OutputThread::createMS()
    {
      // even the HDF5 writer accesses casacore, to perform conversions
      ScopedLock sl(casacoreMutex);
      ScopedDelayCancellation dc; // don't cancel casacore calls

      std::string directoryName = itsTargetDirectory == "" ? itsParset.getDirectoryName(itsOutputType, itsStreamNr) : itsTargetDirectory;
      std::string fileName = itsParset.getFileName(itsOutputType, itsStreamNr);
      std::string path = directoryName + "/" + fileName;

      recursiveMakeDir(directoryName, itsLogPrefix);
      LOG_INFO_STR(itsLogPrefix << "Writing to " << path);

      try {
        // HDF5 writer requested
        switch (itsOutputType) {
        case CORRELATED_DATA:
          itsWriter = new MSWriterCorrelated(itsLogPrefix, path, itsParset, itsStreamNr, itsIsBigEndian);
          break;

#ifdef HAVE_DAL
        case BEAM_FORMED_DATA:
          itsWriter = new MSWriterDAL<float,3>(path, itsParset, itsStreamNr, itsIsBigEndian);
          break;
#endif

        default:
          itsWriter = new MSWriterFile(path);
          break;
        }
      } catch (Exception &ex) {
        LOG_ERROR_STR(itsLogPrefix << "Cannot open " << path << ": " << ex);
        itsWriter = new MSWriterNull;
      }

      // log some core characteristics for CEPlogProcessor for feedback to MoM/LTA
      switch (itsOutputType) {
      case CORRELATED_DATA:
        itsNrExpectedBlocks = itsParset.nrCorrelatedBlocks();

        LOG_INFO_STR(itsLogPrefix << "Characteristics: "
                                  << "SAP " << itsParset.settings.subbands[itsStreamNr].SAP
                                  << ", subband " << itsParset.settings.subbands[itsStreamNr].stationIdx
                                  << ", centralfreq " << setprecision(8) << itsParset.settings.subbands[itsStreamNr].centralFrequency / 1e6 << " MHz"
                                  << ", duration " << setprecision(8) << itsNrExpectedBlocks * itsParset.IONintegrationTime() << " s"
                                  << ", integration " << setprecision(8) << itsParset.IONintegrationTime() << " s"
                                  << ", channels " << itsParset.nrChannelsPerSubband()
                                  << ", channelwidth " << setprecision(8) << itsParset.channelWidth() / 1e3 << " kHz"
                     );
        break;
      case BEAM_FORMED_DATA:
        itsNrExpectedBlocks = itsParset.nrBeamFormedBlocks();
        break;

      default:
        break;
      }
    }


    void OutputThread::checkForDroppedData(StreamableData *data)
    {
      // TODO: check for dropped data at end of observation

      unsigned droppedBlocks = data->sequenceNumber() - itsNextSequenceNumber;

      if (droppedBlocks > 0) {
        itsBlocksDropped += droppedBlocks;

        LOG_WARN_STR(itsLogPrefix << "OutputThread dropped " << droppedBlocks << (droppedBlocks == 1 ? " block" : " blocks"));
      }

      itsNextSequenceNumber = data->sequenceNumber() + 1;
      itsBlocksWritten++;
    }


    static Semaphore writeSemaphore(300);


    void OutputThread::doWork()
    {
      time_t prevlog = 0;

      for (SmartPtr<StreamableData> data; (data = itsReceiveQueue.remove()) != 0; itsFreeQueue.append(data.release())) {
        //NSTimer writeTimer("write data", false, false);

        //writeTimer.start();
        writeSemaphore.down();

        try {
          itsWriter->write(data);
          checkForDroppedData(data);
        } catch (SystemCallException &ex) {
          LOG_WARN_STR(itsLogPrefix << "OutputThread caught non-fatal exception: " << ex.what());
        } catch (...) {
          writeSemaphore.up();
          throw;
        }

        writeSemaphore.up();
        //writeTimer.stop();

        time_t now = time(0L);

        if (now > prevlog + 5) {
          // print info every 5 seconds
          LOG_INFO_STR(itsLogPrefix << "Written block with seqno = " << data->sequenceNumber() << ", " << itsBlocksWritten << " blocks written (" << itsWriter->percentageWritten() << "%), " << itsBlocksDropped << " blocks dropped");

          prevlog = now;
        } else {
          // print debug info for the other blocks
          LOG_DEBUG_STR(itsLogPrefix << "Written block with seqno = " << data->sequenceNumber() << ", " << itsBlocksWritten << " blocks written (" << itsWriter->percentageWritten() << "%), " << itsBlocksDropped << " blocks dropped");
        }
      }
    }


    void OutputThread::cleanUp() const
    {
      float dropPercent = itsBlocksWritten + itsBlocksDropped == 0 ? 0.0 : (100.0 * itsBlocksDropped) / (itsBlocksWritten + itsBlocksDropped);

      LOG_INFO_STR(itsLogPrefix << "Finished writing: " << itsBlocksWritten << " blocks written (" << itsWriter->percentageWritten() << "%), " << itsBlocksDropped << " blocks dropped: " << std::setprecision(3) << dropPercent << "% lost" );
    }


    ParameterSet OutputThread::feedbackLTA() const
    {
      string prefix;

      switch (itsOutputType) {
      case CORRELATED_DATA:
        prefix = formatString("LOFAR.ObsSW.Observation.DataProducts.Output_Correlated_[%u].", itsStreamNr);
        break;

      case BEAM_FORMED_DATA:
        prefix = formatString("LOFAR.ObsSW.Observation.DataProducts.Output_Beamformed_[%u].", itsStreamNr);
        break;

      default:
        prefix = "UNKNOWN.";
        break;
      }

      ParameterSet result;
      result.adoptCollection(itsWriter->configuration(), prefix);

      return result;
    }


    void OutputThread::augment( const FinalMetaData &finalMetaData )
    {
      // wait for writer thread to finish, so we'll have an itsWriter
      ASSERT(itsThread.get());

      itsThread = 0;

      // augment the data product
      ASSERT(itsWriter.get());

      itsWriter->augment(finalMetaData);
    }


    void OutputThread::mainLoop()
    {
      LOG_DEBUG_STR(itsLogPrefix << "OutputThread::mainLoop() entered");

      try {
        createMS();
        doWork();
#if defined HAVE_AIPSPP
      } catch (casa::AipsError &ex) {
        LOG_ERROR_STR(itsLogPrefix << "Caught AipsError: " << ex.what());
        cleanUp();
#endif
      } catch (...) {
        cleanUp(); // Of course, C++ does not need "finally" >:(
        throw;
      }

      cleanUp();
    }

  } // namespace Cobalt
} // namespace LOFAR

