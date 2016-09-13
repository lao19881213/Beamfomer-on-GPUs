//#  OldOutputSection.h: Collects data from CNs and sends data to Storage
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
//#  $Id: OutputSection.h 20457 2012-03-15 15:16:54Z mol $

#ifndef LOFAR_IONPROC_OUTPUT_SECTION_H
#define LOFAR_IONPROC_OUTPUT_SECTION_H

#include <Interface/OutputTypes.h>
#include <Interface/Parset.h>
#include <Interface/SmartPtr.h>
#include <IONProc/OutputThread.h>
#include <Stream/Stream.h>
#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Thread.h>

#include <vector>
#include <string>


namespace LOFAR {
namespace RTCP {

class OutputSection
{
  public:
                                           ~OutputSection();

    void                                   start();

    void				   addIterations(unsigned count);
    void				   noMoreIterations();

  protected:
					   OutputSection(const Parset &, OutputType, unsigned firstBlockNumber, const std::vector<unsigned> &cores, int psetIndex, bool integratable, bool variableNrSubbands);

  private:

    void                                   readData(Stream *, StreamableData *, unsigned steamNr);

    void				   mainLoop();

    void				   droppingData(unsigned subband);
    void				   notDroppingData(unsigned subband);

    const std::string              	   itsLogPrefix;
    const bool                             itsVariableDataSize;
    const Transpose2                       &itsTranspose2Logic;

    const unsigned			   itsNrComputeCores;
    const unsigned			   itsNrCoresPerIteration, itsNrCoresSkippedPerIteration, itsFirstStreamNr, itsNrStreams;
    unsigned				   itsCurrentComputeCore;

    const unsigned			   itsNrIntegrationSteps;
    unsigned				   itsCurrentIntegrationStep;
    const unsigned			   itsNrSamplesPerIntegration;
    unsigned				   itsSequenceNumber;

    const bool                		   itsIsRealTime;
    std::vector<unsigned>		   itsDroppedCount; // [subband]
    std::vector<unsigned>		   itsTotalDroppedCount; // [subband]
    std::vector<SmartPtr<OutputThread> >   itsOutputThreads; // [subband]

    std::vector<SmartPtr<Stream> >	   itsStreamsFromCNs;

    std::vector<SmartPtr<StreamableData> > itsSums;
    SmartPtr<StreamableData>		   itsTmpSum;

    Semaphore				   itsNrIterationsToDo;

    SmartPtr<Thread>			   itsThread;

    std::vector<unsigned>		   itsAdders; // [subband]
};


class PhaseTwoOutputSection : public OutputSection
{
  protected:
    PhaseTwoOutputSection(const Parset &, OutputType, unsigned firstBlockNumber, bool integratable);
};


class PhaseThreeOutputSection : public OutputSection
{
  protected:
    PhaseThreeOutputSection(const Parset &, OutputType, unsigned firstBlockNumber);
};


class CorrelatedDataOutputSection : public PhaseTwoOutputSection
{
  public:
    CorrelatedDataOutputSection(const Parset &, unsigned firstBlockNumber);
};


class BeamFormedDataOutputSection : public PhaseThreeOutputSection
{
  public:
    BeamFormedDataOutputSection(const Parset &, unsigned firstBlockNumber);
};


class TriggerDataOutputSection : public PhaseThreeOutputSection
{
  public:
    TriggerDataOutputSection(const Parset &, unsigned firstBlockNumber);
};


}
}

#endif
