//# BeamFormerSubbandProc.h
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
//# $Id: BeamFormerSubbandProc.h 27477 2013-11-21 13:08:20Z loose $

#ifndef LOFAR_GPUPROC_CUDA_BEAM_FORMER_SUBBAND_PROC_H
#define LOFAR_GPUPROC_CUDA_BEAM_FORMER_SUBBAND_PROC_H

#include <complex>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <CoInterface/StreamableData.h>

#include <GPUProc/gpu_wrapper.h>

#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/Pipelines/BeamFormerPipeline.h>

#include <GPUProc/Kernels/IntToFloatKernel.h>
#include <GPUProc/Kernels/FFT_Kernel.h>
#include <GPUProc/Kernels/DelayAndBandPassKernel.h>
#include <GPUProc/Kernels/BandPassCorrectionKernel.h>
#include <GPUProc/Kernels/BeamFormerKernel.h>
#include <GPUProc/Kernels/BeamFormerTransposeKernel.h>
#include <GPUProc/Kernels/CoherentStokesKernel.h>
#include <GPUProc/Kernels/IncoherentStokesKernel.h>
#include <GPUProc/Kernels/IncoherentStokesTransposeKernel.h>
#include <GPUProc/Kernels/FIR_FilterKernel.h>

#include "SubbandProc.h"

namespace LOFAR
{
  namespace Cobalt
  {
    //# Forward declarations
    struct BeamFormerFactories;

    // Our output data type
    class BeamFormedData : public MultiDimArrayHostBuffer<float, 3>,
                           public StreamableData
    {
    public:
      BeamFormedData(unsigned nrStokes, unsigned nrChannels,
                     size_t nrSamples, gpu::Context &context);
    private:
      virtual void readData(Stream *str, unsigned);
      virtual void writeData(Stream *str, unsigned);
    };

    class BeamFormerSubbandProc : public SubbandProc
    {
    public:
      BeamFormerSubbandProc(const Parset &parset, gpu::Context &context,
                            BeamFormerFactories &factories,
                            size_t nrSubbandsPerSubbandProc = 1);

      // Beam form the data found in the input data buffer
      virtual void processSubband(SubbandProcInputData &input,
                                  StreamableData &output);

      // Do post processing on the CPU
      virtual void postprocessSubband(StreamableData &output);

      // first FFT
      static const size_t DELAY_COMPENSATION_NR_CHANNELS = 64;

      // second FFT
      static const size_t BEAM_FORMER_NR_CHANNELS = 4096;

      // Beamformer specific collection of PerformanceCounters
      class Counters
      {
      public:
        Counters(gpu::Context &context);

        // gpu kernel counters
        PerformanceCounter intToFloat;
        PerformanceCounter firstFFT;
        PerformanceCounter delayBp;
        PerformanceCounter secondFFT;
        PerformanceCounter correctBandpass;
        PerformanceCounter beamformer;
        PerformanceCounter transpose;
        PerformanceCounter inverseFFT;
        PerformanceCounter firFilterKernel;
        PerformanceCounter finalFFT;
        PerformanceCounter coherentStokes;

        PerformanceCounter incoherentInverseFFT;
        PerformanceCounter incoherentFirFilterKernel;
        PerformanceCounter incoherentFinalFFT;
        PerformanceCounter incoherentStokes;
        PerformanceCounter incoherentStokesTranspose;


        // gpu transfer counters
        PerformanceCounter samples;
        PerformanceCounter visibilities;
        PerformanceCounter incoherentOutput;
        // Print the mean and std of each performance counter on the logger
        void printStats();
      };

      Counters counters;

    private:
      // The previously processed SAP/block, or -1 if nothing has been
      // processed yet. Used in order to determine if new delays have
      // to be uploaded.
      ssize_t prevBlock;
      signed int prevSAP;

      // Raw buffers, these are mapped with boost multiarrays 
      // in the InputData class
      SubbandProcInputData::DeviceBuffers devInput;

      // @{
      // Device memory buffers. These buffers are used interleaved.
      gpu::DeviceMemory devA;
      gpu::DeviceMemory devB;
      gpu::DeviceMemory devC;
      gpu::DeviceMemory devD;
      gpu::DeviceMemory devE;
      // @}

      // NULL placeholder for unused DeviceMemory parameters
      gpu::DeviceMemory devNull;

      /*
       * Kernels
       */

      // Int -> Float conversion
      IntToFloatKernel::Buffers intToFloatBuffers;
      std::auto_ptr<IntToFloatKernel> intToFloatKernel;

      // First (64 points) FFT
      FFT_Kernel firstFFT;

      // Delay compensation
      DelayAndBandPassKernel::Buffers delayCompensationBuffers;
      std::auto_ptr<DelayAndBandPassKernel> delayCompensationKernel;

      // Second (64 points) FFT
      FFT_Kernel secondFFT;

      // Bandpass correction and tranpose
      gpu::DeviceMemory devBandPassCorrectionWeights;
      BandPassCorrectionKernel::Buffers bandPassCorrectionBuffers;
      std::auto_ptr<BandPassCorrectionKernel> bandPassCorrectionKernel;

      // *****************************************************************
      //  Objects needed to produce Coherent stokes output
      // beam former

      const bool outputComplexVoltages;
      const bool coherentStokesPPF;

      gpu::DeviceMemory devBeamFormerDelays;
      BeamFormerKernel::Buffers beamFormerBuffers;
      std::auto_ptr<BeamFormerKernel> beamFormerKernel;

      // Transpose 
      BeamFormerTransposeKernel::Buffers transposeBuffers;
      std::auto_ptr<BeamFormerTransposeKernel> transposeKernel;

      // inverse (4k points) FFT
      FFT_Kernel inverseFFT;

      // Poly-phase filter (FIR + FFT)
      gpu::DeviceMemory devFilterWeights;
      gpu::DeviceMemory devFilterHistoryData;
      FIR_FilterKernel::Buffers firFilterBuffers;
      std::auto_ptr<FIR_FilterKernel> firFilterKernel;
      FFT_Kernel finalFFT;

      // Coherent Stokes
      CoherentStokesKernel::Buffers coherentStokesBuffers;
      std::auto_ptr<CoherentStokesKernel> coherentStokesKernel;

      // *****************************************************************
      //  Objects needed to produce incoherent stokes output

      const bool incoherentStokesPPF;

      // Transpose 
      IncoherentStokesTransposeKernel::Buffers incoherentTransposeBuffers;
      std::auto_ptr<IncoherentStokesTransposeKernel> incoherentTranspose;

      // Inverse (4k points) FFT
      FFT_Kernel incoherentInverseFFT;

      // Poly-phase filter (FIR + FFT)
      gpu::DeviceMemory devIncoherentFilterWeights;
      gpu::DeviceMemory devIncoherentFilterHistoryData;
      FIR_FilterKernel::Buffers incoherentFirFilterBuffers;
      std::auto_ptr<FIR_FilterKernel> incoherentFirFilterKernel;
      FFT_Kernel incoherentFinalFFT;

      // Incoherent Stokes
      IncoherentStokesKernel::Buffers incoherentStokesBuffers;
      std::auto_ptr<IncoherentStokesKernel> incoherentStokesKernel;

      bool coherentBeamformer; // TODO temporary hack to allow typing of subband proc
    };

  }
}

#endif

