//# TABTranspose.h
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
//# $Id: BlockID.h 26419 2013-09-09 11:19:56Z mol $

#ifndef LOFAR_COINTERFACE_TABTRANSPOSE_H
#define LOFAR_COINTERFACE_TABTRANSPOSE_H

#include <iostream>
#include <map>
#include <cstring>
#include <Common/Thread/Mutex.h>
#include <Common/Thread/Thread.h>
#include <Stream/Stream.h>
#include <Stream/PortBroker.h>
#include <Common/Thread/Condition.h>
#include <Common/Thread/Mutex.h>
#include "BestEffortQueue.h"
#include "MultiDimArray.h"
#include "SmartPtr.h"
#include "Pool.h"

namespace LOFAR
{
  namespace Cobalt
  {
    namespace TABTranspose
    {
      /*
       * A piece of data belonging to a certain subband.
       *
       * The constructor fixes the size of the data, but
       * other fields are left to the caller to fill.
       */

      struct Subband {
        MultiDimArray<float, 2> data; // [samples][channels]

        Subband( size_t nrSamples = 0, size_t nrChannels = 0 );

        struct {
          size_t fileIdx;
          size_t subband;
          size_t block;
        } id;

        void write(Stream &stream) const;
        void read(Stream &stream);
      };

      /*
       * A block of data, representing for one time slice all
       * subbands.
       *
       * The constructor fixes the size of the data and the number
       * of subbands per block, but other fields are left to the
       * caller to fill.
       */
      class Block {
      public:
        MultiDimArray<float, 3> data; // [subband][samples][channels]
        std::vector<bool> subbandWritten;

        size_t fileIdx;
        size_t block;

        Block( size_t nrSubbands, size_t nrSamples, size_t nrChannels );

        /*
         * Add data for a single subband.
         */
        void addSubband( const Subband &subband );

        /*
         * Zero data of subbands that weren't added.
         */
        void zeroRemainingSubbands();

        /*
         * Return whether the block is complete, that is, all
         * subbands have been added.
         */
        bool complete() const;

      private:
        // The number of subbands left to receive.
        size_t nrSubbandsLeft;
      };

      /*
       * BlockCollector fills Block objects out of individual Subband
       * objects. Fresh blocks are drawn from outputPool.free, and collected
       * blocks are stored in outputPool.filled.
       *
       * The BlockCollector can have several blocks under construction (but
       * no more than maxBlocksInFlight, if >0). Block `b' is emitted to
       * outputPool.filled in ANY of the following cases:
       *
       *   a. all subbands for block 'b' have been received
       *   b. a block younger than 'b' is emitted, which means that
       *      subbands have to arrive in-order to prevent data loss
       *   c. if maxBlocksInFlight > 0, and ALL of the following holds:
       *      - there are maxBlocksInFlight blocks under construction
       *      - a new block is required to store new subbands
       *      - block 'b' is the oldest block
       *   d. finish() is called, which flushes all blocks
       *
       * The finish() call ends by placing a NULL marker in outputPool.filled
       * to indicate the end-of-stream.
       *
       * This class is thread safe.
       */
      class BlockCollector {
      public:
        BlockCollector( Pool<Block> &outputPool, size_t fileIdx, size_t maxBlocksInFlight = 0 );

	/*
         * Add a subband of any block.
         */
        void addSubband( const Subband &subband );

        /*
         * Send all remaining blocks downstream,
         * followed by the end-of-stream marker.
         */
        void finish();

      private:
        std::map<size_t, SmartPtr<Block> > blocks;
        Pool<Block> &outputPool;
        const size_t fileIdx;
        Mutex mutex;

        // upper limit for blocks.size(), or 0 if unlimited
        const size_t maxBlocksInFlight;

        // whether we are allowed to drop data
        const bool canDrop;
        
        // nr of last emitted block, or -1 if no block has been emitted
        ssize_t lastEmitted;

        // The oldest block in flight.
        size_t minBlock() const;

        // The youngest block in flight.
        size_t maxBlock() const;

        /*
         * Send a certain block downstream.
         */
        void emit(size_t blockIdx);

        /*
         * Send all blocks downstream up to
         * and including `block', in-order, as the
         * writer will expect.
         */
        void emitUpTo(size_t block);

        /*
         * Do we manage a certain block?
         */
        bool have(size_t block) const;

        /*
         * Fetch a new block.
         */
        void fetch(size_t block);
      };

      /*
       * Reads multiplexed Subband objects from a stream, and
       * forwards them to set of BlockCollectors. The reception
       * is done in a separate thread.
       */
      class Receiver {
      public:
        // [fileIdx] -> BlockCollector
        typedef std::map<size_t, SmartPtr<BlockCollector> > CollectorMap;

        /*
         * Start receiving from `stream', into `collectors'.
         */
        Receiver( Stream &stream, CollectorMap &collectors );

        // Calls kill()
        ~Receiver();

        // Kills the receiver thread.
        void kill();

        /*
         * Waits for the stream to disconnect and the queue to empty.
         *
         * Returns: true if the receiver thread raised an exception.
         */
        bool finish();

      private:
        Stream &stream;
        CollectorMap &collectors;
        Thread thread;

        void receiveLoop();
      };

      /*
       * MultiReceiver listens on a PortBroker port for multiple streams,
       * dispatching a Receiver for each one.
       */
      class MultiReceiver {
      public:
        MultiReceiver( const std::string &servicePrefix, Receiver::CollectorMap &collectors );

        // Calls kill(0)
        ~MultiReceiver();

        // Kills the listening thread and all client threads.
        //
        // minNrClients: Minimum number of clients to wait to connect and finish.
        //               if minNrClients > 0, all running connections are allowed to
        //               finish, also those beyond minNrClients.
        void kill(size_t minNrClients);

      private:
        Mutex mutex;
        Condition newClient;

        struct Client {
          SmartPtr<PortBroker::ServerStream> stream;
          SmartPtr<Receiver> receiver;
        };

        const std::string servicePrefix;
        Receiver::CollectorMap &collectors;
        std::vector<struct Client> clients;
        Thread thread;

        void listenLoop();

        void dispatch( PortBroker::ServerStream *stream );
      };

      /*
       * MultiSender sends data to various receivers.
       */

      class MultiSender {
      public:
        // A host to send data to, that is, enough information
        // to connect to the PortBroker of the receiver.
        struct Host {
          std::string hostName;
          uint16 brokerPort;
          std::string service;

          bool operator==(const struct Host &other) const {
            return hostName == other.hostName && brokerPort == other.brokerPort && service == other.service;
          };
          bool operator<(const struct Host &other) const {
            if (hostName != other.hostName)
              return hostName < other.hostName;

            if (brokerPort != other.brokerPort)
              return brokerPort < other.brokerPort;

            return service < other.service;
          };
        };

        typedef std::map<size_t,struct Host> HostMap; // fileIdx -> host

        MultiSender( const HostMap &hostMap, size_t queueSize = 3, bool canDrop = false );

        // Send the data from the queues to the receiving hosts. Will run until
        // 'finish()' is called.
        void process();

        // Add a subband for sending. Ownership of the data is taken.
        void append( SmartPtr<struct Subband> &subband );

        // Flush the queues.
        void finish();

      protected:
        // fileIdx -> host mapping
        const HostMap hostMap;

        // Set of hosts to connect to (the list of unique values in hostMap)
        std::vector<struct Host> hosts;

        // A queue for data to be sent to each host
	      std::map<struct Host, SmartPtr< BestEffortQueue< SmartPtr<struct Subband> > > > queues;
      };

    } // namespace TABTranspose
  } // namespace Cobalt
} // namespace LOFAR

#endif

