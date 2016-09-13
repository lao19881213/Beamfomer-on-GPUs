//# BeamFormerFactories.cc
//#
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
//# $Id: BeamFormerFactories.cc 27477 2013-11-21 13:08:20Z loose $

#include <lofar_config.h>

#include "BeamFormerFactories.h"
#include "BeamFormerSubbandProc.h"

namespace LOFAR
{
  namespace Cobalt
  {
    BeamFormerFactories::BeamFormerFactories(const Parset &ps,
                                             size_t nrSubbandsPerSubbandProc) :
        intToFloat(ps),
        delayCompensation(delayCompensationParams(ps)),
        beamFormer(beamFormerParams(ps)),
        transpose(transposeParams(ps)),
        firFilter(firFilterParams(ps, nrSubbandsPerSubbandProc)),
        coherentStokes(coherentStokesParams(ps)),
        incoherentStokes(incoherentStokesParams(ps)),
        incoherentStokesTranspose(incoherentStokesTransposeParams(ps)),
        incoherentFirFilter(
          incoherentFirFilterParams(ps, nrSubbandsPerSubbandProc)),
        bandPassCorrection(bandPassCorrectionParams(ps))
      {
      }

      BandPassCorrectionKernel::Parameters
      BeamFormerFactories::bandPassCorrectionParams(const Parset &ps) const
      {
        BandPassCorrectionKernel::Parameters params(ps);
        params.nrChannels1 =
          BeamFormerSubbandProc::DELAY_COMPENSATION_NR_CHANNELS;
        params.nrChannels2 =
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS /
          params.nrChannels1;
        params.nrSamplesPerChannel = 
          ps.nrSamplesPerSubband() / (params.nrChannels1 * params.nrChannels2);

        return params;
      }

      DelayAndBandPassKernel::Parameters
      BeamFormerFactories::delayCompensationParams(const Parset &ps) const
      {
        DelayAndBandPassKernel::Parameters params(ps);
        params.nrChannelsPerSubband =
          BeamFormerSubbandProc::DELAY_COMPENSATION_NR_CHANNELS;
        params.nrSamplesPerChannel =
          ps.nrSamplesPerSubband() /
          BeamFormerSubbandProc::DELAY_COMPENSATION_NR_CHANNELS;
        params.correctBandPass = false;
        params.transpose = false;

        return params;
      }

      BeamFormerKernel::Parameters 
      BeamFormerFactories::beamFormerParams(const Parset &ps) const
      {
        BeamFormerKernel::Parameters params(ps);
        params.nrChannelsPerSubband =
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS;
        params.nrSamplesPerChannel =
          ps.nrSamplesPerSubband() /
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS;

        return params;
      }

      BeamFormerTransposeKernel::Parameters
      BeamFormerFactories::transposeParams(const Parset &ps) const
      {
        BeamFormerTransposeKernel::Parameters params(ps);
        params.nrChannelsPerSubband =
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS;
        params.nrSamplesPerChannel =
          ps.nrSamplesPerSubband() /
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS;

        return params;
      }

      FIR_FilterKernel::Parameters
      BeamFormerFactories::
      firFilterParams(const Parset &ps,
                      size_t nrSubbandsPerSubbandProc) const
      {
        FIR_FilterKernel::Parameters params(ps);

        params.nrSTABs = ps.settings.beamFormer.maxNrTABsPerSAP();

        // define at least 16 channels to get the FIR_Filter.cu to compile, even
        // if we won't use it.
        params.nrChannelsPerSubband =
          std::max(16U, ps.settings.beamFormer.coherentSettings.nrChannels);

        // time integration has not taken place yet, so calculate the nrSamples
        // manually
        params.nrSamplesPerChannel =
          ps.nrSamplesPerSubband() / params.nrChannelsPerSubband;

        params.nrSubbands = nrSubbandsPerSubbandProc;

        return params;
      }

      CoherentStokesKernel::Parameters
      BeamFormerFactories::coherentStokesParams(const Parset &ps) const
      {
        CoherentStokesKernel::Parameters params(ps);
        params.nrChannelsPerSubband =
          ps.settings.beamFormer.coherentSettings.nrChannels;
        params.nrSamplesPerChannel =
          ps.nrSamplesPerSubband() / params.nrChannelsPerSubband;

        return params;
      }

      FIR_FilterKernel::Parameters 
      BeamFormerFactories::
      incoherentFirFilterParams(const Parset &ps,
            size_t nrSubbandsPerSubbandProc) const 
      {
        FIR_FilterKernel::Parameters params(ps);

        params.nrSTABs = ps.nrStations();

        params.nrChannelsPerSubband = 
          ps.settings.beamFormer.incoherentSettings.nrChannels;

        // time integration has not taken place yet, so calculate the nrSamples
        // manually
        params.nrSamplesPerChannel = 
          ps.nrSamplesPerSubband() / params.nrChannelsPerSubband;

        params.nrSubbands = nrSubbandsPerSubbandProc;

        return params;
      }

      IncoherentStokesKernel::Parameters 
      BeamFormerFactories::
      incoherentStokesParams(const Parset &ps) const 
      {
        IncoherentStokesKernel::Parameters params(ps);
        params.nrChannelsPerSubband = 
          ps.settings.beamFormer.incoherentSettings.nrChannels;
        params.nrSamplesPerChannel = 
          ps.nrSamplesPerSubband() / params.nrChannelsPerSubband;

        return params;
      }

      IncoherentStokesTransposeKernel::Parameters 
      BeamFormerFactories::
      incoherentStokesTransposeParams(const Parset &ps) const 
      {
        IncoherentStokesTransposeKernel::Parameters params(ps);
        params.nrChannelsPerSubband =
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS;
        params.nrSamplesPerChannel =
          ps.nrSamplesPerSubband() /
          BeamFormerSubbandProc::BEAM_FORMER_NR_CHANNELS;

        return params;
      }
  }
}
