//# CorrelatorSubbandProc.cc
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
//# $Id: CorrelatorWorkQueue.cc 27386 2013-11-13 16:55:14Z amesfoort $

#include <lofar_config.h>

#include "CorrelatorSubbandProc.h"

#include <cstring>
#include <algorithm>

#include <Common/LofarLogger.h>

#include <GPUProc/OpenMP_Lock.h>
#include <GPUProc/BandPass.h>
#include <GPUProc/Pipelines/CorrelatorPipelinePrograms.h>

namespace LOFAR
{
  namespace Cobalt
  {
    /* The data travels as follows:
     *
     * [input]  -> devInput.inputSamples
     *             -> firFilterKernel
     *          -> devFilteredData
     *             -> fftKernel
     *          -> devFilteredData
     *             -> delayAndBandPassKernel
     *          -> devInput.inputSamples
     *             -> correlatorKernel
     *          -> devFilteredData = visibilities
     * [output] <-
     */
    CorrelatorSubbandProc::CorrelatorSubbandProc(const Parset       &parset,
      cl::Context &context, 
      cl::Device  &device,
      unsigned gpuNumber,
                                             CorrelatorPipelinePrograms & programs,
                                             FilterBank &filterBank
                                             )
      :
    SubbandProc( context, device, gpuNumber, parset),
      prevBlock(-1),
      prevSAP(-1),
      devInput(ps.nrBeams(),
                ps.nrStations(),
                NR_POLARIZATIONS,
                ps.nrHistorySamples() + ps.nrSamplesPerSubband(),
                ps.nrBytesPerComplexSample(),
                queue,

                // reserve enough space in inputSamples for the output of
                // the delayAndBandPassKernel.
                ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerSubband() * sizeof(std::complex<float>)),
      devFilteredData(queue,
                      CL_MEM_READ_WRITE,

                      // reserve enough space for the output of the
                      // firFilterKernel,
                      std::max(ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerSubband() * sizeof(std::complex<float>),
                      // and the correlatorKernel.
                               ps.nrBaselines() * ps.nrChannelsPerSubband() * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>))),
      devFIRweights(queue,
                    CL_MEM_READ_ONLY,
                    ps.nrChannelsPerSubband() * NR_TAPS * sizeof(float)),
      firFilterKernel(ps,
                      queue,
                      programs.firFilterProgram,
                      devFilteredData,
                      devInput.inputSamples,
                      devFIRweights),
      fftKernel(ps,
                context,
                devFilteredData),
      bandPassCorrectionWeights(boost::extents[ps.nrChannelsPerSubband()],
                                queue,
                                CL_MEM_WRITE_ONLY,
                                CL_MEM_READ_ONLY),
      delayAndBandPassKernel(ps,
                             programs.delayAndBandPassProgram,
                             devInput.inputSamples,
                             devFilteredData,
                             devInput.delaysAtBegin,
                             devInput.delaysAfterEnd,
                             devInput.phaseOffsets,
                             bandPassCorrectionWeights),
#if defined USE_NEW_CORRELATOR
      correlateTriangleKernel(ps,
                              queue,
                              programs.correlatorProgram,
                              devFilteredData,
                              devInput.inputSamples),
      correlateRectangleKernel(ps,
                              queue,
                              programs.correlatorProgram, 
                              devFilteredData, 
                              devInput.inputSamples)
#else
      correlatorKernel(ps,
                       queue, 
                       programs.correlatorProgram, 
                       devFilteredData, 
                       devInput.inputSamples)
#endif
    {
      // put enough objects in the inputPool to operate
      // TODO: Tweak the number of inputPool objects per SubbandProc,
      // probably something like max(3, nrSubbands/nrSubbandProcs * 2), because
      // there both need to be enough items to receive all subbands at
      // once, and enough items to process the same amount in the
      // mean time.
      //
      // At least 3 items are needed for a smooth Pool operation.
      size_t nrInputDatas = std::max(3UL, ps.nrSubbands());
      for(size_t i = 0; i < nrInputDatas; ++i) {
        inputPool.free.append(new SubbandProcInputData(
                ps.nrBeams(),
                ps.nrStations(),
                NR_POLARIZATIONS,
                ps.nrHistorySamples() + ps.nrSamplesPerSubband(),
                ps.nrBytesPerComplexSample(),
                devInput));
      }

      // put enough objects in the outputPool to operate
      for(size_t i = 0; i < 3; ++i) {
        outputPool.free.append(new CorrelatedDataHostBuffer(
                ps.nrStations(),
                ps.nrChannelsPerSubband(),
                ps.integrationSteps(),
                devFilteredData,
                *this));
      }

      // create all the counters
      // Move the FIR filter weight to the GPU
#if defined USE_NEW_CORRELATOR
      addCounter("compute - cor.triangle");
      addCounter("compute - cor.rectangle");
#else
      addCounter("compute - correlator");
#endif

      addCounter("compute - FIR");
      addCounter("compute - delay/bp");
      addCounter("compute - FFT");
      addCounter("input - samples");
      addCounter("output - visibilities");

      // CPU timers are set by CorrelatorPipeline
      addTimer("CPU - read input");
      addTimer("CPU - process");
      addTimer("CPU - postprocess");
      addTimer("CPU - total");

      // GPU timers are set by us
      addTimer("GPU - total");
      addTimer("GPU - input");
      addTimer("GPU - output");
      addTimer("GPU - compute");
      addTimer("GPU - wait");

      queue.enqueueWriteBuffer(devFIRweights, CL_TRUE, 0, ps.nrChannelsPerSubband() * NR_TAPS * sizeof(float), filterBank.getWeights().origin());

      if (ps.correctBandPass())
      {
        BandPass::computeCorrectionFactors(bandPassCorrectionWeights.origin(), ps.nrChannelsPerSubband());
        bandPassCorrectionWeights.hostToDevice(CL_TRUE);
      }
    }

    // Get the log2 of the supplied number
    unsigned CorrelatorSubbandProc::flagFunctions::get2LogOfNrChannels(unsigned nrChannels)
    {
      ASSERT(powerOfTwo(nrChannels));

      unsigned logNrChannels;
      for (logNrChannels = 0; 1U << logNrChannels != nrChannels;
        logNrChannels ++)
      {;} // do nothing, the creation of the log is a side effect of the for loop

      //Alternative solution snipped:
      //int targetlevel = 0;
      //while (index >>= 1) ++targetlevel; 
      return logNrChannels;
    }

    void CorrelatorSubbandProc::flagFunctions::propagateFlagsToOutput(
      Parset const &parset,
      MultiDimArray<LOFAR::SparseSet<unsigned>, 1>const &inputFlags,
      CorrelatedData &output)
    {   
      unsigned numberOfChannels = parset.nrChannelsPerSubband();

      // Object for storing transformed flags
      MultiDimArray<SparseSet<unsigned>, 2> flagsPerChannel(
        boost::extents[numberOfChannels][parset.nrStations()]);

      // First transform the flags to channel flags: taking in account 
      // reduced resolution in time and the size of the filter
      convertFlagsToChannelFlags(parset, inputFlags, flagsPerChannel);

      // Calculate the number of flafs per baseline and assign to
      // output object.
      switch (output.itsNrBytesPerNrValidSamples) {
        case 4:
          calculateAndSetNumberOfFlaggedSamples<uint32_t>(parset, flagsPerChannel, output);
          break;

        case 2:
          calculateAndSetNumberOfFlaggedSamples<uint16_t>(parset, flagsPerChannel, output);
          break;

        case 1:
          calculateAndSetNumberOfFlaggedSamples<uint8_t>(parset, flagsPerChannel, output);
          break;
      }
    }

    void CorrelatorSubbandProc::flagFunctions::convertFlagsToChannelFlags(Parset const &parset,
      MultiDimArray<LOFAR::SparseSet<unsigned>, 1>const &inputFlags,
      MultiDimArray<SparseSet<unsigned>, 2>& flagsPerChannel)
    {
      unsigned numberOfChannels = parset.nrChannelsPerSubband();
      unsigned log2NrChannels = get2LogOfNrChannels(numberOfChannels);
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
          if (numberOfChannels == 1)  // if number of channels == 1
          { //do nothing, just take the ranges as supplied
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


    namespace {
      unsigned baseline(unsigned stat1, unsigned stat2)
      {
        //baseline(stat1, stat2); This function should be moved to a helper class
        return stat2 * (stat2 + 1) / 2 + stat1;
      }
    }

    template<typename T> void CorrelatorSubbandProc::flagFunctions::calculateAndSetNumberOfFlaggedSamples(
      Parset const &parset,
      MultiDimArray<SparseSet<unsigned>, 2>const & flagsPerChannel,
      CorrelatedData &output)
    {
      // loop the stations
      for (unsigned stat2 = 0; stat2 < parset.nrStations(); stat2 ++) {
        for (unsigned stat1 = 0; stat1 <= stat2; stat1 ++) {
          unsigned bl = baseline(stat1, stat2);

          unsigned nrSamplesPerIntegration = parset.nrSamplesPerChannel();
          // If there is a single channel then the index 0 contains real data
          if (parset.nrChannelsPerSubband() == 1) 
          {                                            
            //The number of invalid (flagged) samples is the union of the flagged samples in the two stations
            unsigned nrValidSamples = nrSamplesPerIntegration -
              (flagsPerChannel[0][stat1] | flagsPerChannel[0][stat2]).count();

            // Moet worden toegekend op de correlated dataobject
            output.nrValidSamples<T>(bl, 0) = nrValidSamples;
          } 
          else 
          {
            // channel 0 does not contain valid data
            output.nrValidSamples<T>(bl, 0) = 0; //channel zero, has zero valid samples

            for(unsigned ch = 1; ch < parset.nrChannelsPerSubband(); ch ++) 
            {
              // valid samples is total number of samples minus the union of the
              // Two stations.
              unsigned nrValidSamples = nrSamplesPerIntegration -
                (flagsPerChannel[ch][stat1] | flagsPerChannel[ch][stat2]).count();

              output.nrValidSamples<T>(bl, ch) = nrValidSamples;
            }
          }
        }
      }
    }

    // Instantiate required templates
    template void CorrelatorSubbandProc::flagFunctions::calculateAndSetNumberOfFlaggedSamples<uint32_t>(
      Parset const &parset,
      MultiDimArray<SparseSet<unsigned>, 2>const & flagsPerChannel,
      CorrelatedData &output);
    template void CorrelatorSubbandProc::flagFunctions::calculateAndSetNumberOfFlaggedSamples<uint16_t>(
      Parset const &parset,
      MultiDimArray<SparseSet<unsigned>, 2>const & flagsPerChannel,
      CorrelatedData &output);
    template void CorrelatorSubbandProc::flagFunctions::calculateAndSetNumberOfFlaggedSamples<uint8_t>(
      Parset const &parset,
      MultiDimArray<SparseSet<unsigned>, 2>const & flagsPerChannel,
      CorrelatedData &output);

    void CorrelatorSubbandProc::flagFunctions::applyWeightingToAllPolarizations(unsigned baseline, 
      unsigned channel, float weight, CorrelatedData &output)
    { // TODO: inline???
      for(unsigned idx_polarization_1 = 0; idx_polarization_1 < NR_POLARIZATIONS; ++idx_polarization_1)
        for(unsigned idx_polarization_2 = 0; idx_polarization_2 < NR_POLARIZATIONS; ++idx_polarization_2)
          output.visibilities[baseline][channel][idx_polarization_1][idx_polarization_2] *= weight;
    }

    template<typename T> void CorrelatorSubbandProc::flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities(Parset const &parset,
      CorrelatedData &output)
    {
      for (unsigned bl = 0; bl < output.itsNrBaselines; ++bl) {
        // Calculate the weights for the channels
        //
        // Channel 0 is already flagged according to specs, so we can simply
        // include it both for 1 and >1 channels/subband.
        for(unsigned ch = 0; ch < parset.nrChannelsPerSubband(); ch ++) 
        {
          T nrValidSamples = output.nrValidSamples<T>(bl, ch);

          // If all samples flagged weights is zero
          // TODO: make a lookup table for the expensive division
          float weight = nrValidSamples ? 1e-6f / nrValidSamples : 0;  

          applyWeightingToAllPolarizations(bl, ch, weight, output);
        }
      }
    }

    // Instantiate required templates
    template void CorrelatorSubbandProc::flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities<uint32_t>(Parset const &parset,
      CorrelatedData &output);
    template void CorrelatorSubbandProc::flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities<uint16_t>(Parset const &parset,
      CorrelatedData &output);
    template void CorrelatorSubbandProc::flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities<uint8_t>(Parset const &parset,
      CorrelatedData &output);


    void CorrelatorSubbandProc::processSubband(SubbandProcInputData &input, CorrelatedDataHostBuffer &output)
    {
      timers["GPU - total"]->start();

      size_t block = input.block;
      unsigned subband = input.subband;

      {
        timers["GPU - input"]->start();

#if defined USE_B7015
        OMP_ScopedLock scopedLock(pipeline.hostToDeviceLock[gpu / 2]);
#endif
        input.inputSamples.hostToDevice(CL_TRUE);
        counters["input - samples"]->doOperation(input.inputSamples.deviceBuffer.event, 0, 0, input.inputSamples.bytesize());

        timers["GPU - input"]->stop();
      }

      timers["GPU - compute"]->start();

      // Moved from doWork() The delay data should be available before the kernels start.
      // Queue processed ordered. This could main that the transfer is not nicely overlapped

      unsigned SAP = ps.settings.subbands[subband].SAP;

      // Only upload delays if they changed w.r.t. the previous subband
      if ((int)SAP != prevSAP || (ssize_t)block != prevBlock) {
        input.delaysAtBegin.hostToDevice(CL_FALSE);
        input.delaysAfterEnd.hostToDevice(CL_FALSE);
        input.phaseOffsets.hostToDevice(CL_FALSE);

        prevSAP = SAP;
        prevBlock = block;
      }

      if (ps.nrChannelsPerSubband() > 1) {
        firFilterKernel.enqueue(queue, *counters["compute - FIR"]);
        fftKernel.enqueue(queue, *counters["compute - FFT"]);
      }

      delayAndBandPassKernel.enqueue(queue, *counters["compute - delay/bp"], subband);
#if defined USE_NEW_CORRELATOR
      correlateTriangleKernel.enqueue(queue, *counters["compute - cor.triangle"]);
      correlateRectangleKernel.enqueue(queue, *counters["compute - cor.rectangle"]);
#else
      correlatorKernel.enqueue(queue, *counters["compute - correlator"]);
#endif

      queue.flush();

      // ***** The GPU will be occupied for a while, do some calculations in the
      // background.

      // Propagate the flags.
      flagFunctions::propagateFlagsToOutput(ps, input.inputFlags, output);

      // Wait for the GPU to finish.
      timers["GPU - wait"]->start();
      queue.finish();
      timers["GPU - wait"]->stop();

      timers["GPU - compute"]->stop();

      {
        timers["GPU - output"]->start();

#if defined USE_B7015
        OMP_ScopedLock scopedLock(pipeline.deviceToHostLock[gpu / 2]);
#endif
        output.deviceToHost(CL_TRUE);
        // now perform weighting of the data based on the number of valid samples

        counters["output - visibilities"]->doOperation(output.deviceBuffer.event, 0, output.bytesize(), 0);

        timers["GPU - output"]->stop();
      }

      timers["GPU - total"]->stop();
    }


    void CorrelatorSubbandProc::postprocessSubband(CorrelatedDataHostBuffer &output)
    {
      // The flags are alrady copied to the correct location
      // now the flagged amount should be applied to the visibilities
      switch (output.itsNrBytesPerNrValidSamples) {
        case 4:
          flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities<uint32_t>(ps, output);  
          break;

        case 2:
          flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities<uint16_t>(ps, output);  
          break;

        case 1:
          flagFunctions::applyFractionOfFlaggedSamplesOnVisibilities<uint8_t>(ps, output);  
          break;
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

  }
}

