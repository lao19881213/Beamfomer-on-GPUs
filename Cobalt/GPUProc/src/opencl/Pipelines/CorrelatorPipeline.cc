//# CorrelatorPipeline.cc
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
//# $Id: CorrelatorPipeline.cc 25727 2013-07-22 12:56:14Z mol $

#include <lofar_config.h>

#include "CorrelatorPipeline.h"

#include <iomanip>

#include <Common/LofarLogger.h>
#include <ApplCommon/PosixTime.h>
#include <Stream/Stream.h>
#include <Stream/FileStream.h>
#include <Stream/NullStream.h>
#include <CoInterface/CorrelatedData.h>
#include <CoInterface/Stream.h>

#include <InputProc/SampleType.h>
#include <InputProc/Transpose/MPIReceiveStations.h>

#include <GPUProc/OpenMP_Lock.h>
#include <GPUProc/SubbandProcs/CorrelatorSubbandProc.h>
#include <GPUProc/SubbandProcs/SubbandProc.h>

using namespace std;

namespace LOFAR
{
  namespace Cobalt
  {

    CorrelatorPipeline::CorrelatorPipeline(const Parset &ps)
      :
      Pipeline(ps),
      subbandPool(ps.nrSubbands()),
      filterBank(true, NR_TAPS, ps.nrChannelsPerSubband(), KAISER)
    {
      filterBank.negateWeights();
      double startTime = omp_get_wtime();

      //#pragma omp parallel sections
      {
        programs.firFilterProgram = createProgram("FIR.cl");
        programs.delayAndBandPassProgram = createProgram("DelayAndBandPass.cl");
#if defined USE_NEW_CORRELATOR
        programs.correlatorProgram = createProgram("NewCorrelator.cl");
#else
        programs.correlatorProgram = createProgram("Correlator.cl");
#endif
      }
      LOG_DEBUG_STR("compile time = " << omp_get_wtime() - startTime);
    }

    void CorrelatorPipeline::doWork()
    {
      size_t nrSubbandProcs = (profiling ? 1 : 2) * nrGPUs;
      vector< SmartPtr<CorrelatorSubbandProc> > workQueues(nrSubbandProcs);

      for (size_t i = 0; i < workQueues.size(); ++i) {
        workQueues[i] = new CorrelatorSubbandProc(ps,               // Configuration
                                      context,          // Opencl context
                                      devices[i % nrGPUs], // The GPU this workQueue is connected to
                                      i % nrGPUs, // The GPU index
                                      programs,         // The compiled kernels, const
                                      filterBank);   // The filter set to use. Const
      }

      for (unsigned sb = 0; sb < ps.nrSubbands(); sb++) {
        // Allow 10 blocks to be in the best-effort queue.
        // TODO: make this dynamic based on memory or time
        subbandPool[sb].bequeue = new BestEffortQueue< SmartPtr<CorrelatedDataHostBuffer> >(3, ps.realTime());
      }

      double startTime = ps.startTime();
      double stopTime = ps.stopTime();
      double blockTime = ps.CNintegrationTime();

      size_t nrBlocks = floor((stopTime - startTime) / blockTime);

      //sections = program segments defined by the following omp section directive
      //           are distributed for parallel execution among available threads
      //parallel = directive explicitly instructs the compiler to parallelize the chosen block of code.
      //  The two sections in this function are done in parallel with a seperate set of threads.
#     pragma omp parallel sections
      {
        /*
         * BLOCK OF SUBBANDS -> WORKQUEUE INPUTPOOL
         */
#       pragma omp section
        {
          switch (ps.nrBitsPerSample()) {
          default:
          case 16:
            receiveInput< SampleType<i16complex> >(nrBlocks, workQueues);
            break;
          case 8:
            receiveInput< SampleType<i8complex> >(nrBlocks, workQueues);
            break;
          case 4:
            receiveInput< SampleType<i4complex> >(nrBlocks, workQueues);
            break;
          }
        }


        /*
         * WORKQUEUE INPUTPOOL -> WORKQUEUE OUTPUTPOOL
         *
         * Perform GPU processing, one thread per workQueue.
         */
#       pragma omp section
        {
#         pragma omp parallel for num_threads(workQueues.size())
          for (size_t i = 0; i < workQueues.size(); ++i) {
            CorrelatorSubbandProc &queue = *workQueues[i];

            // run the queue
            queue.timers["CPU - total"]->start();
            processSubbands(queue);
            queue.timers["CPU - total"]->stop();

            // Signal end of output
            queue.outputPool.filled.append(NULL);
          }
        }

        /*
         * WORKQUEUE OUTPUTPOOL -> SUBBANDPOOL
         *
         * Perform post-processing, one thread per workQueue.
         */
#       pragma omp section
        {
#         pragma omp parallel for num_threads(workQueues.size())
          for (size_t i = 0; i < workQueues.size(); ++i) {
            CorrelatorSubbandProc &queue = *workQueues[i];

            // run the queue
            postprocessSubbands(queue);
          }

          // Signal end of output
          for (size_t subband = 0; subband < ps.nrSubbands(); ++subband) {
            subbandPool[subband].bequeue->noMore();
          }
        }

        /*
         * SUBBANDPOOL -> STORAGE STREAMS (best effort)
         */
#       pragma omp section
        {
#         pragma omp parallel for num_threads(ps.nrSubbands())
          for (size_t subband = 0; subband < ps.nrSubbands(); ++subband) {
            // write subband to Storage
            writeSubband(subband);
          }
        }
      }

      // gather performance figures
      for (size_t i = 0; i < workQueues.size(); ++i ) {
        performance.addQueue(*workQueues[i]);
      }

      // log performance figures
      performance.log(workQueues.size());
    }


    // Record type needed by receiveInput. Before c++0x, a local type
    // can't be a template argument, so we'll have to define this type
    // globally.
    struct inputData_t {
      // An InputData object suited for storing one subband from all
      // stations.
      SmartPtr<SubbandProcInputData> data;

      // The SubbandProc associated with the data
      CorrelatorSubbandProc *queue;
    };

    template<typename SampleT> void CorrelatorPipeline::receiveInput( size_t nrBlocks, const std::vector< SmartPtr<CorrelatorSubbandProc> > &workQueues )
    {
      // The length of a block in samples
      size_t blockSize = ps.nrHistorySamples() + ps.nrSamplesPerSubband();

      // SEND: For now, the n stations are sent by the first n ranks.
      vector<int> stationRanks(ps.nrStations());
      for (size_t stat = 0; stat < ps.nrStations(); ++stat) {
        stationRanks[stat] = stat;
      }

      // RECEIVE: For now, we receive ALL beamlets.
      vector<size_t> subbands(ps.nrSubbands());
      for (size_t subband = 0; subband < ps.nrSubbands(); ++subband) {
        subbands[subband] = subband;
      }

      // Set up the MPI environment.
      MPIReceiveStations receiver(stationRanks, subbands, blockSize);

      // Create a block object to hold all information for receiving one
      // block.
      vector<struct MPIReceiveStations::Block<SampleT> > blocks(ps.nrStations());

      for (size_t stat = 0; stat < ps.nrStations(); ++stat) {
        blocks[stat].beamlets.resize(ps.nrSubbands());
      }

      size_t workQueueIterator = 0;

      for (size_t block = 0; block < nrBlocks; block++) {
        // Receive the samples of all subbands from the stations for this
        // block.

        // The set of InputData objects we're using for this block.
        vector<struct inputData_t> inputDatas(ps.nrSubbands());

        for (size_t subband = 0; subband < ps.nrSubbands(); ++subband) {
          // Fetch an input object to store this subband. For now, blindly
          // round-robin over the work queues.
          CorrelatorSubbandProc &queue = *workQueues[workQueueIterator++ % workQueues.size()];

          // Fetch an input object to fill from the selected queue.
          // NOTE: We'll put it in a SmartPtr right away!
          SmartPtr<SubbandProcInputData> data = queue.inputPool.free.remove();

          // Annotate the block
          data->block   = block;
          data->subband = subband;

          // Incorporate it in the receiver's input set.
          for (size_t stat = 0; stat < ps.nrStations(); ++stat) {
            blocks[stat].beamlets[subband].samples = reinterpret_cast<SampleT*>(&data->inputSamples[stat][0][0][0]);
          }

          // Record the block (transfers ownership)
          inputDatas[subband].data = data;
          inputDatas[subband].queue = &queue;
        }

        // Receive all subbands from all stations
        LOG_INFO_STR("[block " << block << "] Reading input samples");
        receiver.receiveBlock<SampleT>(blocks);

        // Process and forward the received input to the processing threads
        for (size_t subband = 0; subband < ps.nrSubbands(); ++subband) {
          CorrelatorSubbandProc &queue = *inputDatas[subband].queue;
          SmartPtr<SubbandProcInputData> data = inputDatas[subband].data;

          // Translate the metadata as provided by receiver
          for (size_t stat = 0; stat < ps.nrStations(); ++stat) {
            SubbandMetaData &metaData = blocks[stat].beamlets[subband].metaData;

            // extract and apply the flags
            // TODO: Not in this thread! Add a preprocess thread maybe?
            data->inputFlags[stat] = metaData.flags;

            data->flagInputSamples(stat, metaData);

            // extract and assign the delays for the station beams
            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++)
            {
              unsigned sap = ps.settings.subbands[subband].SAP;

              data->delaysAtBegin[sap][stat][pol] = metaData.stationBeam.delayAtBegin;
              data->delaysAfterEnd[sap][stat][pol] = metaData.stationBeam.delayAfterEnd;
              data->phaseOffsets[stat][pol] = 0.0;
            }
          }

          queue.inputPool.filled.append(data);
        }

        LOG_DEBUG_STR("[block " << block << "] Forwarded input to processing");
      }

      // Signal end of input
      for (size_t i = 0; i < workQueues.size(); ++i) {
        workQueues[i]->inputPool.filled.append(NULL);
      }
    }

    template void CorrelatorPipeline::receiveInput< SampleType<i16complex> >( size_t nrBlocks, const std::vector< SmartPtr<CorrelatorSubbandProc> > &workQueues );
    template void CorrelatorPipeline::receiveInput< SampleType<i8complex> >( size_t nrBlocks, const std::vector< SmartPtr<CorrelatorSubbandProc> > &workQueues );
    template void CorrelatorPipeline::receiveInput< SampleType<i4complex> >( size_t nrBlocks, const std::vector< SmartPtr<CorrelatorSubbandProc> > &workQueues );


    void CorrelatorPipeline::processSubbands(CorrelatorSubbandProc &workQueue)
    {
      SmartPtr<SubbandProcInputData> input;

      // Keep fetching input objects until end-of-input
      while ((input = workQueue.inputPool.filled.remove()) != NULL) {
        size_t block = input->block;
        unsigned subband = input->subband;

        if (subband == 0 || subband == ps.nrSubbands() - 1) {
          LOG_INFO_STR("[block " << block << ", subband " << subband << "] Processing start");
        }

        // Also fetch an output object to store results
        SmartPtr<CorrelatedDataHostBuffer> output = workQueue.outputPool.free.remove();
        ASSERT(output != NULL); // Only we signal end-of-data, so we should never receive it

        output->block = block;
        output->subband = subband;

        // Perform calculations
        workQueue.timers["CPU - process"]->start();
        workQueue.processSubband(*input, *output);
        workQueue.timers["CPU - process"]->stop();

        // Hand off output to post processing
        workQueue.outputPool.filled.append(output);
        ASSERT(!output);

        // Give back input data for a refill
        workQueue.inputPool.free.append(input);
        ASSERT(!input);

        if (subband == 0 || subband == ps.nrSubbands() - 1) {
          LOG_DEBUG_STR("[block " << block << ", subband " << subband << "] Forwarded output to post processing");
        }
      }
    }


    void CorrelatorPipeline::postprocessSubbands(CorrelatorSubbandProc &workQueue)
    {
      SmartPtr<CorrelatedDataHostBuffer> output;

      size_t nrBlocksForwarded = 0;
      size_t nrBlocksDropped = 0;
      time_t lastLogTime = 0;

      // Keep fetching output objects until end-of-output
      while ((output = workQueue.outputPool.filled.remove()) != NULL) {
        size_t block = output->block;
        unsigned subband = output->subband;

        if (subband == 0 || subband == ps.nrSubbands() - 1) {
          LOG_INFO_STR("[block " << block << ", subband " << subband << "] Post processing start");
        }

        workQueue.timers["CPU - postprocess"]->start();
        workQueue.postprocessSubband(*output);
        workQueue.timers["CPU - postprocess"]->stop();

        // Hand off output, force in-order as Storage expects it that way
        subbandPool[subband].sync.waitFor(block);

        // We do the ordering, so we set the sequence numbers
        output->setSequenceNumber(block);

        if (!subbandPool[subband].bequeue->append(output)) {
          nrBlocksDropped++;
          //LOG_WARN_STR("[block " << block << "] Dropped for subband " << subband);

          // Give back to queue
          workQueue.outputPool.free.append(output);
        } else {
          nrBlocksForwarded++;
        }

        // Allow next block to be written
        subbandPool[subband].sync.advanceTo(block + 1);

        ASSERT(!output);

        if (subband == 0 || subband == ps.nrSubbands() - 1) {
          LOG_DEBUG_STR("[block " << block << ", subband " << subband << "] Forwarded output to writer");
        }

        if (time(0) != lastLogTime) {
          lastLogTime = time(0);

          LOG_INFO_STR("Forwarded " << nrBlocksForwarded << " blocks, dropped " << nrBlocksDropped << " blocks");
        }
      }
    }


    void CorrelatorPipeline::writeSubband( unsigned subband )
    {
      SmartPtr<Stream> outputStream;

      // Connect to output stream
      try {
        if (ps.getHostName(CORRELATED_DATA, subband) == "") {
          // an empty host name means 'write to disk directly', to
          // make debugging easier for now
          outputStream = new FileStream(ps.getFileName(CORRELATED_DATA, subband), 0666);
        } else {
          // connect to the Storage_main process for this output
          const std::string desc = getStreamDescriptorBetweenIONandStorage(ps, CORRELATED_DATA, subband);

          outputStream = createStream(desc, false, 0);
        }
      } catch(Exception &ex) {
        LOG_ERROR_STR("Dropping rest of subband " << subband << ": " << ex);

        outputStream = new NullStream;
      }

      SmartPtr<CorrelatedDataHostBuffer> output;

      // Process pool elements until end-of-output
      while ((output = subbandPool[subband].bequeue->remove()) != NULL) {
        size_t block = output->block;
        unsigned subband = output->subband;

        CorrelatorSubbandProc &queue = output->queue; // cache queue object, because `output' will be destroyed

        if (subband == 0 || subband == ps.nrSubbands() - 1) {
          LOG_INFO_STR("[block " << block << ", subband " << subband << "] Writing start");
        }

        // Write block to disk 
        try {
          output->write(outputStream.get(), true);
        } catch(Exception &ex) {
          LOG_ERROR_STR("Dropping rest of subband " << subband << ": " << ex);

          outputStream = new NullStream;
        }

        // Hand the object back to the workQueue it originally came from
        queue.outputPool.free.append(output);

        ASSERT(!output);

        if (subband == 0 || subband == ps.nrSubbands() - 1) {
          LOG_INFO_STR("[block " << block << ", subband " << subband << "] Done");
        }
      }
    }

  }
}

