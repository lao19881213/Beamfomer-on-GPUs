//# SubbandProc.cc
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: SubbandProc.cc 27386 2013-11-13 16:55:14Z amesfoort $

#include <lofar_config.h>

#include "SubbandProc.h"

#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    SubbandProc::SubbandProc(const Parset &ps, gpu::Context &context, size_t nrSubbandsPerSubbandProc)
    :
      ps(ps),
      nrSubbandsPerSubbandProc(nrSubbandsPerSubbandProc),
      queue(gpu::Stream(context))
    {
      // put enough objects in the inputPool to operate
      //
      // At least 3 items are needed for a smooth Pool operation.
      size_t nrInputDatas = std::max(3UL, 2 * nrSubbandsPerSubbandProc);
      for (size_t i = 0; i < nrInputDatas; ++i) {
        inputPool.free.append(new SubbandProcInputData(
                ps.nrBeams(),
                ps.nrStations(),
                ps.settings.nrPolarisations,
                ps.settings.beamFormer.maxNrTABsPerSAP(),
                ps.nrSamplesPerSubband(),
                ps.nrBytesPerComplexSample(),
                context));
      }
    }

    SubbandProc::~SubbandProc()
    {
    }

    void SubbandProc::addTimer(const std::string &name)
    {
      timers[name] = new NSTimer(name, false, false);
    }


    size_t SubbandProc::nrOutputElements() const
    {
      /*
       * Output elements can get stuck in:
       *   Best-effort queue:       3 elements
       *   In flight to outputProc: 1 element
       *
       * which means we'll need at least 5 elements
       * in the pool to get a smooth operation.
       */
      return 5 * nrSubbandsPerSubbandProc;
    }


    void SubbandProcInputData::applyMetaData(const Parset &ps,
                                           unsigned station, unsigned SAP,
                                           const SubbandMetaData &metaData)
    {
      // extract and apply the flags
      inputFlags[station] = metaData.flags;

      flagInputSamples(station, metaData);

      // extract and assign the delays for the station beams

      // X polarisation
      delaysAtBegin[SAP][station][0]  = ps.settings.stations[station].delayCorrection.x + metaData.stationBeam.delayAtBegin;
      delaysAfterEnd[SAP][station][0] = ps.settings.stations[station].delayCorrection.x + metaData.stationBeam.delayAfterEnd;
      phaseOffsets[station][0]        = ps.settings.stations[station].phaseCorrection.x;

      // Y polarisation
      delaysAtBegin[SAP][station][1]  = ps.settings.stations[station].delayCorrection.y + metaData.stationBeam.delayAtBegin;
      delaysAfterEnd[SAP][station][1] = ps.settings.stations[station].delayCorrection.y + metaData.stationBeam.delayAfterEnd;
      phaseOffsets[station][1]        = ps.settings.stations[station].phaseCorrection.y;


      if (ps.settings.beamFormer.enabled)
      {
        for (unsigned tab = 0; tab < metaData.TABs.size(); tab++)
        {
          // we already compensated for the delay for the first beam
          double compensatedDelay = (metaData.stationBeam.delayAfterEnd +
                                     metaData.stationBeam.delayAtBegin) * 0.5;

          // subtract the delay that was already compensated for
          tabDelays[SAP][station][tab] = (metaData.TABs[tab].delayAtBegin +
                                          metaData.TABs[tab].delayAfterEnd) * 0.5 -
                                         compensatedDelay;
        }
      }
    }


    // flag the input samples.
    void SubbandProcInputData::flagInputSamples(unsigned station,
                                              const SubbandMetaData& metaData)
    {

      // Get the size of a sample in bytes.
      size_t sizeof_sample = sizeof *inputSamples.origin();

      // Calculate the number elements to skip when striding over the second
      // dimension of inputSamples.
      size_t stride = inputSamples[station][0].num_elements();

      // Zero the bytes in the input data for the flagged ranges.
      for(SparseSet<unsigned>::const_iterator it = metaData.flags.getRanges().begin();
        it != metaData.flags.getRanges().end(); ++it)
      {
        void *offset = inputSamples[station][it->begin].origin();
        size_t size = stride * (it->end - it->begin) * sizeof_sample;
        memset(offset, 0, size);
      }
    }


    // Get the log2 of the supplied number
    // TODO: move this into a util/helper function/file (just like CorrelatorSubbandProc.cc::baseline() and Align.h::powerOfTwo(),nextPowerOfTwo())
    unsigned SubbandProc::Flagger::log2(unsigned n)
    {
      ASSERT(powerOfTwo(n));

      unsigned log;
      for (log = 0; 1U << log != n; log ++)
        {;} // do nothing, the creation of the log is a side effect of the for loop

      //Alternative solution snipped:
      //int targetlevel = 0;
      //while (index >>= 1) ++targetlevel; 
      return log;
    }

    void SubbandProc::Flagger::convertFlagsToChannelFlags(Parset const &parset,
      MultiDimArray<LOFAR::SparseSet<unsigned>, 1>const &inputFlags,
      MultiDimArray<SparseSet<unsigned>, 2>& flagsPerChannel)
    {
      unsigned numberOfChannels = parset.nrChannelsPerSubband();
      unsigned log2NrChannels = log2(numberOfChannels);
      //Convert the flags per sample to flags per channel
      for (unsigned station = 0; station < parset.nrStations(); station ++) 
      {
        // get the flag ranges
        const SparseSet<unsigned>::Ranges &ranges = inputFlags[station].getRanges();
        for (SparseSet<unsigned>::const_iterator it = ranges.begin();
          it != ranges.end(); it ++) 
        {
          unsigned begin_idx;
          unsigned end_idx;
          if (numberOfChannels == 1)
          {
            // do nothing, just take the ranges as supplied
            begin_idx = it->begin; 
            end_idx = std::min(parset.nrSamplesPerChannel(), it->end );
          }
          else
          {
            // Never flag before the start of the time range               
            // use bitshift to divide to the number of channels. 
            //
            // NR_TAPS is the width of the filter: they are
            // absorbed by the FIR and thus should be excluded
            // from the original flag set.
            //
            // At the same time, every sample is affected by
            // the NR_TAPS-1 samples before it. So, any flagged
            // sample in the input flags NR_TAPS samples in
            // the channel.
            begin_idx = std::max(0, 
              (signed) (it->begin >> log2NrChannels) - NR_TAPS + 1);

            // The min is needed, because flagging the last input
            // samples would cause NR_TAPS subsequent samples to
            // be flagged, which aren't necessarily part of this block.
            end_idx = std::min(parset.nrSamplesPerChannel() + 1, 
              ((it->end - 1) >> log2NrChannels) + 1);
          }

          // Now copy the transformed ranges to the channelflags
          for (unsigned ch = 0; ch < numberOfChannels; ch++) {
            flagsPerChannel[ch][station].include(begin_idx, end_idx);
          }
        }
      }
    }
  }
}

