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

#include <BeamletBufferToComputeNode.h>
#include <ControlPhase3Cores.h>
#include <Common/LofarLogger.h>
#include <Stream/PortBroker.h>
#include <Interface/Stream.h>
#include <Interface/CN_Command.h>
#include <Interface/Exceptions.h>
#include <Interface/PrintVector.h>
#include <Interface/RSPTimeStamp.h>
#include <InputSection.h>
#include <ION_Allocator.h>
#include <Scheduling.h>
#include <GlobalVars.h>
#include <Job.h>
#include <Scheduling.h>
#include <OutputSection.h>
#include <StreamMultiplexer.h>
#include <Stream/SocketStream.h>
#include <Stream/PortBroker.h>

#include <unistd.h>
#include <time.h>

#include <boost/format.hpp>


#define LOG_CONDITION (myPsetNumber == 0)

namespace LOFAR {
namespace RTCP {

unsigned Job::nextJobID = 1;
void	 *Job::theInputSection;
Mutex	 Job::theInputSectionMutex;
unsigned Job::theInputSectionRefCount = 0;

Queue<Job *> finishedJobs;


Job::Job(const char *parsetName)
:
  itsParset(parsetName),
  itsJobID(nextJobID ++), // no need to make thread safe
  itsObservationID(itsParset.observationID()),
  itsLogPrefix(str(boost::format("[obs %d] ") % itsParset.observationID())),
  itsIsRunning(false),
  itsDoCancel(false),
  itsBlockNumber(0),
  itsRequestedStopTime(0.0),
  itsStopTime(0.0),
  itsStorageProcesses(itsParset, itsLogPrefix)
{
  // fill the cache to avoid regenerating it many times over
  itsParset.write(NULL);

  if (LOG_CONDITION) {
    // Handle PVSS (CEPlogProcessor) communication -- report PVSS name in the first log line to allow CEPlogProcessor to resolve obsIDs
    if (itsParset.PVSS_TempObsName() != "")
      LOG_INFO_STR(itsLogPrefix << "PVSS name: " << itsParset.PVSS_TempObsName());

    LOG_INFO_STR(itsLogPrefix << "----- Creating new job");
    LOG_DEBUG_STR(itsLogPrefix << "usedCoresInPset = " << itsParset.usedCoresInPset());
  }

  LOG_DEBUG_STR(itsLogPrefix << "Running from " << (unsigned long)itsParset.startTime() << " to " << (unsigned long)itsParset.stopTime());

  // check enough parset settings just to get to the coordinated check in jobThread safely
  if (itsParset.CNintegrationTime() <= 0)
    THROW(IONProcException,"CNintegrationTime must be bigger than 0");

  // synchronize roughly every 5 seconds to see if the job is cancelled
  itsNrBlockTokensPerBroadcast = static_cast<unsigned>(ceil(5.0 / itsParset.CNintegrationTime()));
  itsNrBlockTokens	       = 1; // trigger a rendez-vous immediately to sync latest stoptime info

  itsHasPhaseOne   = itsParset.phaseOnePsetIndex(myPsetNumber) >= 0;
  itsHasPhaseTwo   = itsParset.phaseTwoPsetIndex(myPsetNumber) >= 0;
  itsHasPhaseThree = itsParset.phaseThreePsetIndex(myPsetNumber) >= 0;

  itsJobThread = new Thread(this, &Job::jobThread, itsLogPrefix + "[JobThread] ", 65536);
}


Job::~Job()
{
  // stop any started Storage processes
  if (myPsetNumber == 0)
    itsStorageProcesses.stop(0);

  if (LOG_CONDITION)
    LOG_INFO_STR(itsLogPrefix << "----- Job " << (itsIsRunning ? "finished" : "cancelled") << " successfully");
}


void Job::createIONstreams()
{
  if (myPsetNumber == 0) {
    std::vector<unsigned> involvedPsets = itsParset.usedPsets();

    for (unsigned i = 0; i < involvedPsets.size(); i ++) {
      ASSERT(involvedPsets[i] < allIONstreamMultiplexers.size());

      if (involvedPsets[i] != 0) // do not send to itself
	itsIONstreams.push_back(new MultiplexedStream(*allIONstreamMultiplexers[involvedPsets[i]], itsJobID));
    }    
  } else {
    itsIONstreams.push_back(new MultiplexedStream(*allIONstreamMultiplexers[0], itsJobID));
  }
}


void Job::barrier()
{
  char byte = 0;

  if (myPsetNumber == 0) {
    for (unsigned i = 0; i < itsIONstreams.size(); i ++) {
      itsIONstreams[i]->read(&byte, sizeof byte);
      itsIONstreams[i]->write(&byte, sizeof byte);
    }
  } else {
    itsIONstreams[0]->write(&byte, sizeof byte);
    itsIONstreams[0]->read(&byte, sizeof byte);
  }
}


// returns true iff all psets supply true
bool Job::agree(bool iAgree)
{
  bool allAgree = iAgree; // pset 0 needs to start with its own decision, for other psets this value is ignored

  if (myPsetNumber == 0)
    for (unsigned i = 0; i < itsIONstreams.size(); i ++) {
      bool youAgree;
      itsIONstreams[i]->read(&youAgree, sizeof youAgree);

      allAgree = allAgree && youAgree;
    }
  else
    itsIONstreams[0]->write(&iAgree, sizeof iAgree);

  broadcast(allAgree);  

  return allAgree;
}


template <typename T> void Job::broadcast(T &value)
{
  if (myPsetNumber == 0)
    for (unsigned i = 0; i < itsIONstreams.size(); i ++)
      itsIONstreams[i]->write(&value, sizeof value);
  else
    itsIONstreams[0]->read(&value, sizeof value);
}


void Job::waitUntilCloseToStartOfObservation(time_t secondsPriorToStart)
{
  time_t closeToStart = static_cast<time_t>(itsParset.startTime()) - secondsPriorToStart;
  char   buf[26];

  ctime_r(&closeToStart, buf);
  buf[24] = '\0';
  
  LOG_INFO_STR(itsLogPrefix << "Waiting for job to start: sleeping until " << buf);

  itsWallClockTime.waitUntil(closeToStart);
}


void Job::cancel()
{
  // note that JobQueue holds lock, so that this function executes atomically

  if (itsDoCancel) {
    LOG_WARN_STR(itsLogPrefix << "Observation already cancelled");
  } else {
    LOG_WARN_STR(itsLogPrefix << "Cancelling observation");
    itsDoCancel = true;
    //jobQueue.itsReevaluate.broadcast();

    if (itsParset.realTime())
      itsWallClockTime.cancelWait();
  }
}


void Job::claimResources()
{
  ScopedLock scopedLock(jobQueue.itsMutex);

  while (!itsDoCancel) {
    bool conflict = false;

    for (std::vector<Job *>::iterator job = jobQueue.itsJobs.begin(); job != jobQueue.itsJobs.end(); job ++) {
      std::stringstream error;

      if ((*job)->itsIsRunning && (*job)->itsParset.conflictingResources(itsParset, error)) {
	conflict = true;
	LOG_WARN_STR(itsLogPrefix << "Postponed due to resource conflict with job " << (*job)->itsObservationID << ": " << error.str());
      }
    }

    if (!conflict) {
      itsIsRunning = true;
      return;
    }

    jobQueue.itsReevaluate.wait(jobQueue.itsMutex);
  }
}


void Job::jobThread()
{
#if defined HAVE_BGP_ION
  doNotRunOnCore0();
#endif

  if (myPsetNumber == 0 || itsHasPhaseOne || itsHasPhaseTwo || itsHasPhaseThree) {
    createCNstreams();
    createIONstreams();

    if (myPsetNumber == 0) {
      // DEFINE phase
      bool canStart = true;

      if (!checkParset()) {
        canStart = false;
      }

      // obey the stop time in the parset -- the first startBlock() will broadcast it
      if (!pause(itsParset.stopTime())) {
        LOG_ERROR_STR(itsLogPrefix << "Could not set observation stop time");
        canStart = false;
      }

      if (canStart) {
        // INIT phase
        if (itsParset.realTime())
          waitUntilCloseToStartOfObservation(20);

        claimResources();

        // we could start Storage before claiming resources
        if (itsIsRunning && itsParset.hasStorage())
          itsStorageProcesses.start();
      } 
    }

    broadcast(itsIsRunning);

    if (itsIsRunning) {
      // RUN phase

      if (itsParset.realTime()) {
        // if we started after itsParset.startTime(), we want to skip ahead to
        // avoid data loss caused by having to catch up.
        if (myPsetNumber == 0) {
          time_t earliest_start = time(0L) + 5;

          if (earliest_start > itsParset.startTime()) {
            itsBlockNumber = static_cast<unsigned>((earliest_start - itsParset.startTime()) / itsParset.CNintegrationTime());

            LOG_WARN_STR(itsLogPrefix << "Skipping the first " << itsBlockNumber << " blocks to catch up"); 
          } else {
            itsBlockNumber = 0;
          }  
        }

        broadcast(itsBlockNumber);
      }

      // each node is expected to:
      // 1. agree() on starting, to allow the compute nodes to complain in preprocess()
      // 2. call startBlock() until the end of the observation to synchronise the
      //    stop time.

      if (itsHasPhaseOne || itsHasPhaseTwo || itsHasPhaseThree) {
	switch (itsParset.nrBitsPerSample()) {
	  case  4 : doObservation<i4complex>();
		    break;

	  case  8 : doObservation<i8complex>();
		    break;

	  case 16 : doObservation<i16complex>();
		    break;
	}
      } else {
        if (agree(true)) { // we always agree on the fact that we can start
          // force pset 0 to broadcast itsIsRunning periodically
	  while (startBlock())
	    endBlock();
        }    
      }    

      // PAUSE phase
      barrier();

      // RELEASE phase
      itsIsRunning = false;
      jobQueue.itsReevaluate.broadcast();

      if (myPsetNumber == 0) {
        itsStorageProcesses.forwardFinalMetaData(time(0) + 240);

        // all InputSections and OutputSections have finished their processing, so
        // Storage should be done any second now.

        itsStorageProcesses.stop(time(0) + 300);
      }

      // Augment the LTA feedback logging
      if (myPsetNumber == 0) {
        ParameterSet feedbackLTA;
        feedbackLTA.add("Observation.DataProducts.nrOfOutput_Beamformed_", str(boost::format("%u") % itsParset.nrStreams(BEAM_FORMED_DATA)));
        feedbackLTA.add("Observation.DataProducts.nrOfOutput_Correlated_", str(boost::format("%u") % itsParset.nrStreams(CORRELATED_DATA)));

        for (ParameterSet::const_iterator i = feedbackLTA.begin(); i != feedbackLTA.end(); ++i)
          LOG_INFO_STR(itsLogPrefix << "LTA FEEDBACK: " << i->first << " = " << i->second);
      }  
    }
  }

  finishedJobs.append(this);
}


void Job::createCNstreams()
{
  std::vector<unsigned> usedCoresInPset = itsParset.usedCoresInPset();

  itsCNstreams.resize(usedCoresInPset.size());

  for (unsigned core = 0; core < usedCoresInPset.size(); core ++) {
    ASSERT(usedCoresInPset[core] < nrCNcoresInPset);
    itsCNstreams[core] = allCNstreams[myPsetNumber][usedCoresInPset[core]];
  }

  if (itsHasPhaseOne || itsHasPhaseTwo) {
    std::vector<unsigned> phaseOneTwoCores = itsParset.phaseOneTwoCores();

    itsPhaseOneTwoCNstreams.resize(nrPsets, phaseOneTwoCores.size());

#ifdef CLUSTER_SCHEDULING
    for (unsigned pset = 0; pset < nrPsets; pset ++)
#else
    unsigned pset = myPsetNumber;
#endif
    {
      for (unsigned core = 0; core < phaseOneTwoCores.size(); core ++) {
        ASSERT(phaseOneTwoCores[core] < nrCNcoresInPset);
        itsPhaseOneTwoCNstreams[pset][core] = allCNstreams[pset][phaseOneTwoCores[core]];
      }
    }
  }

  if (itsHasPhaseThree) {
    std::vector<unsigned> phaseThreeCores = itsParset.phaseThreeCores();

    itsPhaseThreeCNstreams.resize(phaseThreeCores.size());

    for (unsigned core = 0; core < phaseThreeCores.size(); core ++) {
      ASSERT(phaseThreeCores[core] < nrCNcoresInPset);
      itsPhaseThreeCNstreams[core] = allCNstreams[myPsetNumber][phaseThreeCores[core]];
    }
  }
}


template <typename SAMPLE_TYPE> void Job::attachToInputSection()
{
  ScopedLock scopedLock(theInputSectionMutex);

  if (theInputSectionRefCount == 0) {
    theInputSection = new InputSection<SAMPLE_TYPE>(itsParset, myPsetNumber);
    ++ theInputSectionRefCount;
  }
}


template <typename SAMPLE_TYPE> void Job::detachFromInputSection()
{
  ScopedLock scopedLock(theInputSectionMutex);

  if (-- theInputSectionRefCount == 0)
    delete static_cast<InputSection<SAMPLE_TYPE> *>(theInputSection);
}


bool Job::configureCNs()
{
  bool success = true;

  CN_Command command(CN_Command::PREPROCESS, itsBlockNumber);
  
  LOG_DEBUG_STR(itsLogPrefix << "Configuring cores " << itsParset.usedCoresInPset() << " ...");

  for (unsigned core = 0; core < itsCNstreams.size(); core ++) {
    command.write(itsCNstreams[core]);
    itsParset.write(itsCNstreams[core]);
  }

#if 0 // FIXME: leads to deadlock when using TCP
  for (unsigned core = 0; core < itsCNstreams.size(); core ++) {
    char failed;
    itsCNstreams[core]->read(&failed, sizeof failed);

    if (failed) {
      LOG_ERROR_STR(itsLogPrefix << "Core " << core << " failed to initialise");
      success = false;
    }
  }
#endif
  
  LOG_DEBUG_STR(itsLogPrefix << "Configuring cores " << itsParset.usedCoresInPset() << " done");

  return success;
}


void Job::unconfigureCNs()
{
  CN_Command command(CN_Command::POSTPROCESS);
 
  LOG_DEBUG_STR(itsLogPrefix << "Unconfiguring cores " << itsParset.usedCoresInPset() << " ...");

  for (unsigned core = 0; core < itsCNstreams.size(); core ++)
    command.write(itsCNstreams[core]);

  LOG_DEBUG_STR(itsLogPrefix << "Unconfiguring cores " << itsParset.usedCoresInPset() << " done");
}


bool Job::startBlock()
{
  if (-- itsNrBlockTokens == 0) {
    itsNrBlockTokens = itsNrBlockTokensPerBroadcast;

    // only consider cancelling at itsNrBlockTokensPerBroadcast boundaries
    itsIsRunning = !itsDoCancel;

    // only allow pset 0 to actually decide whether or not to stop
    broadcast(itsIsRunning);

    // sync updated stop times -- abuse atomicity of copying itsRequestedStopTime
    itsStopTime = itsRequestedStopTime;
    broadcast(itsStopTime);
  }

//  LOG_INFO_STR("anotherRun: itsBlockNumber = " << itsBlockNumber);

  bool done = !itsIsRunning;

  if (itsStopTime > 0.0) {
    // the end time of this block must still be within the observation
    double currentTime = itsParset.startTime() + (itsBlockNumber + 1) * itsParset.CNintegrationTime();

    done = done || currentTime > itsStopTime;
  }

//  LOG_INFO_STR("anotherRun: done = " << done);

  return !done;
}

void Job::endBlock()
{
  itsBlockNumber++;
}


template <typename SAMPLE_TYPE> void Job::doObservation()
{
  std::vector<OutputSection *> outputSections;

  if (LOG_CONDITION)
    LOG_INFO_STR(itsLogPrefix << "----- Observation start");

  // first: send configuration to compute nodes so they know what to expect
  if (!agree(configureCNs())) {
    unconfigureCNs();

    if (LOG_CONDITION)
      LOG_INFO_STR(itsLogPrefix << "----- Observation finished");

    return;
  }

  if (itsHasPhaseOne)
    attachToInputSection<SAMPLE_TYPE>();

  if (itsHasPhaseTwo) {
    if (itsParset.outputCorrelatedData())
      outputSections.push_back(new CorrelatedDataOutputSection(itsParset, itsBlockNumber));
  }

  if (itsHasPhaseThree) {
    if (itsParset.outputBeamFormedData())
      outputSections.push_back(new BeamFormedDataOutputSection(itsParset, itsBlockNumber));

    if (itsParset.outputTrigger())
      outputSections.push_back(new TriggerDataOutputSection(itsParset, itsBlockNumber));
  }

  // start the threads
  for (unsigned i = 0; i < outputSections.size(); i ++)
    outputSections[i]->start();

  LOG_DEBUG_STR(itsLogPrefix << "doObservation processing input start");

  { // separate scope to ensure that the beamletbuffertocomputenode objects
    // only exist if the beamletbuffers exist in the inputsection
    std::vector<SmartPtr<BeamletBuffer<SAMPLE_TYPE> > > noInputs;
    BeamletBufferToComputeNode<SAMPLE_TYPE>   beamletBufferToComputeNode(itsParset, itsPhaseOneTwoCNstreams, itsHasPhaseOne ? static_cast<InputSection<SAMPLE_TYPE> *>(theInputSection)->itsBeamletBuffers : noInputs, myPsetNumber, itsBlockNumber);

    ControlPhase3Cores controlPhase3Cores(itsParset, itsPhaseThreeCNstreams, itsBlockNumber);
    controlPhase3Cores.start(); // start the thread

    while (startBlock()) {
      for (unsigned i = 0; i < outputSections.size(); i ++)
	outputSections[i]->addIterations(1);

      controlPhase3Cores.addIterations(1);

      beamletBufferToComputeNode.process();

      endBlock();
    }

    LOG_DEBUG_STR(itsLogPrefix << "doObservation processing input done");
  }

  for (unsigned i = 0; i < outputSections.size(); i ++)
    outputSections[i]->noMoreIterations();

  for (unsigned i = 0; i < outputSections.size(); i ++)
    delete outputSections[i];

  if (itsHasPhaseOne)
    detachFromInputSection<SAMPLE_TYPE>();

  unconfigureCNs();
 
  if (LOG_CONDITION)
    LOG_INFO_STR(itsLogPrefix << "----- Observation finished");
}


bool Job::checkParset() const
{
  // any error detected by the python environment, invalidating this parset
  string pythonParsetError = itsParset.getString("OLAP.IONProc.parsetError","");

  if (pythonParsetError != "" ) {
    LOG_ERROR_STR(itsLogPrefix << "Early detected parset error: " << pythonParsetError );
    return false;
  }

  try {
    itsParset.check();
  } catch( InterfaceException &ex ) {
    LOG_ERROR_STR(itsLogPrefix << "Parset check failed on " << ex.what() );
    return false;
  }

  if (itsParset.nrCoresPerPset() > nrCNcoresInPset) {
    LOG_ERROR_STR(itsLogPrefix << "nrCoresPerPset (" << itsParset.nrCoresPerPset() << ") cannot exceed " << nrCNcoresInPset);
    return false;
  }

  return true;
}


void Job::printInfo() const
{
  LOG_INFO_STR(itsLogPrefix << "JobID = " << itsJobID << ", " << (itsIsRunning ? "running" : "not running"));
}


bool Job::pause(const double &when)
{
  char   buf[26];
  time_t whenRounded = static_cast<time_t>(when);

  ctime_r(&whenRounded, buf);
  buf[24] = '\0';
  
  LOG_DEBUG_STR(itsLogPrefix << "Job: pause(): pause observation at " << buf);

  // make sure we don't interfere with queue dynamics
  ScopedLock scopedLock(jobQueue.itsMutex);

  if (itsParset.realTime() && (when == 0 || when <= itsParset.startTime())) { // yes we can compare a double to 0
    // make sure we also stop waiting for the job to start

    if (!itsDoCancel)
      cancel();
  } else {
    LOG_WARN_STR(itsLogPrefix << "Non-real time mode blablabla");
    itsRequestedStopTime = when;
  }

  return true;
}


bool Job::quit()
{
  LOG_DEBUG_STR(itsLogPrefix << "Job: quit(): end observation");
  // stop now

  if (!itsDoCancel) {
    ScopedLock scopedLock(jobQueue.itsMutex);

    cancel();
  }

  return true;
}


} // namespace RTCP
} // namespace LOFAR
