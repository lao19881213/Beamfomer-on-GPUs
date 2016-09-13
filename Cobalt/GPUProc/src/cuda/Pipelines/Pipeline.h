//# Pipeline.h
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
//# $Id: Pipeline.h 27103 2013-10-27 09:51:53Z mol $

#ifndef LOFAR_GPUPROC_CUDA_PIPELINE_H
#define LOFAR_GPUPROC_CUDA_PIPELINE_H

#include <string>
#include <vector>
#include <map>

#include <Common/LofarTypes.h>
#include <Common/Thread/Queue.h>
#include <Common/Thread/Mutex.h>
#include <CoInterface/BestEffortQueue.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/SlidingPointer.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/OpenMP_Lock.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/PerformanceCounter.h>
#include <GPUProc/SubbandProcs/SubbandProc.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class Pipeline
    {
    public:
      Pipeline(const Parset &ps, const std::vector<size_t> &subbandIndices, const std::vector<gpu::Device> &devices);

      virtual ~Pipeline();

      // for each subband get data from input stream, sync, start the kernels to process all data, write output in parallel
      void processObservation(OutputType outputType);

    protected:
      const Parset             &ps;
      const std::vector<gpu::Device> devices;

      const std::vector<size_t> subbandIndices; // [localSubbandIdx]
      std::vector< SmartPtr<SubbandProc> > workQueues;

      const size_t nrSubbandsPerSubbandProc;

#if defined USE_B7015
      OMP_Lock hostToDeviceLock[4], deviceToHostLock[4];
#endif

      // Combines all functionality needed for getting the total from a set of
      // counters
      struct Performance
      {
        std::map<std::string, SmartPtr<NSTimer> > total_timers;
        // lock on the shared data
        Mutex totalsMutex;
        // add the counter in this queue
        void addQueue(SubbandProc &queue);
        // Print a logline with results
        void log(size_t nrSubbandProcs);

        size_t nrGPUs;

        Performance(size_t nrGPUs = 1);
      } performance;

    private:
      struct Output {
        // synchronisation to write blocks in-order
        SlidingPointer<size_t> sync;

        // output data queue
        SmartPtr< BestEffortQueue< SmartPtr<StreamableData> > > bequeue;
      };

      // For each block, read all subbands from all stations, and divide the
      // work over the workQueues
      void receiveInput( size_t nrBlocks );

      // Templated version of receiveInput(), to specialise in receiving
      // a certain type of input sample.
      template<typename SampleT> void receiveInput( size_t nrBlocks );

      // preprocess subbands on the CPU
      void preprocessSubbands(SubbandProc &workQueue);

      // process subbands on the GPU
      void processSubbands(SubbandProc &workQueue);

      // Post-process subbands on the CPU
      void postprocessSubbands(SubbandProc &workQueue);

      // Send subbands to Storage
      void writeSubband(unsigned globalSubbandIdx, struct Output &output,
                        SmartPtr<Stream> outputStream);

      // Create Stream to Storage
      SmartPtr<Stream> connectToOutput(unsigned globalSubbandIdx,
                                       OutputType outputType) const;

      std::vector<struct Output> writePool; // [localSubbandIdx]
    };
  }
}

#endif

