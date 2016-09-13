//#  Stream.cc: one line descriptor
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
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
//#  $Id: Stream.cc 16396 2010-09-27 12:12:24Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Interface/BeamFormedData.h>
#include <Interface/CorrelatedData.h>
#include <Interface/DataFactory.h>
#include <Interface/FilteredData.h>
#include <Interface/TriggerData.h>


namespace LOFAR {
namespace RTCP {


StreamableData *newStreamableData(const Parset &parset, OutputType outputType, int streamNr, Allocator &allocator)
{
  switch (outputType) {
    case CORRELATED_DATA   : return new CorrelatedData(parset.nrMergedStations(), parset.nrChannelsPerSubband(), parset.integrationSteps(), allocator);

    case BEAM_FORMED_DATA  : {
      const Transpose2 &beamFormLogic = parset.transposeLogic();

      unsigned nrSubbands    = streamNr == -1 ? beamFormLogic.maxNrSubbands() : beamFormLogic.streamInfo[streamNr].subbands.size();
      unsigned nrChannels    = streamNr == -1 ? beamFormLogic.maxNrChannels() : beamFormLogic.streamInfo[streamNr].nrChannels;
      unsigned nrSamples     = streamNr == -1 ? beamFormLogic.maxNrSamples()  : beamFormLogic.streamInfo[streamNr].nrSamples;

      return new FinalBeamFormedData(nrSamples, nrSubbands, nrChannels, allocator);
    }

    case TRIGGER_DATA      : return new TriggerData;

    default		   : THROW(InterfaceException, "unsupported output type");
  }

}


} // namespace RTCP
} // namespace LOFAR
