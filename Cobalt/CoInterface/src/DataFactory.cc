//# DataFactory.cc
//# Copyright (C) 2011-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: DataFactory.cc 25662 2013-07-13 19:10:59Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <CoInterface/DataFactory.h>

#include <CoInterface/OutputTypes.h>
#include <CoInterface/CorrelatedData.h>
#include <CoInterface/BeamFormedData.h>
#include <CoInterface/TriggerData.h>


namespace LOFAR
{
  namespace Cobalt
  {


    StreamableData *newStreamableData(const Parset &parset, OutputType outputType, unsigned streamNr, Allocator &allocator)
    {
      switch (outputType) {
      case CORRELATED_DATA: return new CorrelatedData(parset.nrMergedStations(), parset.nrChannelsPerSubband(), parset.integrationSteps(), allocator, 512); // 512 alignment is needed for writing to MS

      case BEAM_FORMED_DATA: {
        const struct ObservationSettings::BeamFormer::File &file = parset.settings.beamFormer.files[streamNr];
        const struct ObservationSettings::BeamFormer::StokesSettings &sset =
          file.coherent ? parset.settings.beamFormer.coherentSettings
                        : parset.settings.beamFormer.incoherentSettings;

        unsigned nrSubbands = parset.settings.nrSubbands(file.sapNr);
        unsigned nrChannels = sset.nrChannels;
        unsigned nrSamples = parset.settings.nrSamplesPerSubband() / sset.nrChannels / sset.timeIntegrationFactor;

        return new FinalBeamFormedData(nrSamples, nrSubbands, nrChannels, allocator);
      }

      case TRIGGER_DATA: return new TriggerData;

      default: THROW(CoInterfaceException, "unsupported output type");
      }

    }


  } // namespace Cobalt
} // namespace LOFAR

