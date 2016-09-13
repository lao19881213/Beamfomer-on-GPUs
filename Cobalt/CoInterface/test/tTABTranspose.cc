//# tParset.cc
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
//# $Id: tParset.cc 25931 2013-08-05 13:07:35Z klijn $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <Stream/StringStream.h>
#include <CoInterface/TABTranspose.h>

#include <UnitTest++.h>
#include <boost/format.hpp>
#include <omp.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace LOFAR::Cobalt::TABTranspose;
using namespace std;
using boost::format;

SUITE(Block) {
  TEST(OrderedArrival) {
    const size_t nrSubbands = 10;
    const size_t nrSamples = 1024;
    const size_t nrChannels = 64;

    Block block(nrSubbands, nrSamples, nrChannels);

    for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
      Subband subband(nrSamples, nrChannels);
      subband.id.subband = subbandIdx;

      CHECK(!block.complete());
      block.addSubband(subband);
    }

    CHECK(block.complete());
  }

  TEST(UnorderedArrival) {
    const size_t nrSubbands = 10;
    const size_t nrSamples = 1024;
    const size_t nrChannels = 64;

    Block block(nrSubbands, nrSamples, nrChannels);

    // Our increment needs to be co-prime to nrSubbands for
    // the ring to visit all elements.
    const size_t subbandIncrement = 7;

    CHECK(nrSubbands % subbandIncrement != 0);
    CHECK(subbandIncrement % nrSubbands != 0);

    for (size_t n = 0; n < nrSubbands; ++n) {
      // avoid starting at 0, because the ordered test already does
      size_t subbandIdx = (3 + n * subbandIncrement) % nrSubbands;

      Subband subband(nrSamples, nrChannels);
      subband.id.subband = subbandIdx;

      CHECK(!block.complete());
      block.addSubband(subband);
    }

    CHECK(block.complete());
  }
}


/* A basic BlockCollector setup */
struct Fixture {
  static const size_t nrSubbands = 10;
  static const size_t nrSamples = 1024;
  static const size_t nrChannels = 64;
  static const size_t nrBlocks = 3;

  Pool<Block> outputPool;
  BlockCollector ctr;

  Fixture()
  :
    ctr(outputPool, 0)
  {
    for (size_t i = 0; i < nrBlocks; ++i) {
      outputPool.free.append(new Block(nrSubbands, nrSamples, nrChannels));
    }
  }
};

struct Fixture_Loss: public Fixture {
  static const size_t maxInFlight = 2;

  BlockCollector ctr_loss;

  Fixture_Loss()
  :
    ctr_loss(outputPool, 0, maxInFlight)
  {
    // add some subbands for both blocks
    for (size_t blockIdx = 0; blockIdx < maxInFlight; ++blockIdx) {
      for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
        // We'll drop subband 3.
        //
        // Note that we drop the same subband for all blocks, to prevent
        // early detection of data loss.
        if (subbandIdx == 3)
          continue;

        Subband sb(nrSamples, nrChannels);
        sb.id.block = blockIdx;
        sb.id.subband = subbandIdx;

        ctr_loss.addSubband(sb);
      }
    }

    // shouldn't have emitted anything yet
    CHECK_EQUAL(0UL, outputPool.filled.size());
  }

};

SUITE(BlockCollector) {
  TEST_FIXTURE(Fixture, OneBlock) {
    // add all subbands for block 0
    for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
      Subband sb(nrSamples, nrChannels);
      sb.id.block = 0;
      sb.id.subband = subbandIdx;

      ctr.addSubband(sb);
    }

    CHECK_EQUAL(1UL,          outputPool.filled.size());
    CHECK_EQUAL(nrBlocks - 1, outputPool.free.size());
  }

  TEST_FIXTURE(Fixture, AllBlocks) {
    // add all subbands for all blocks
    for (size_t blockIdx = 0; blockIdx < nrBlocks; ++blockIdx) {
      for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
        Subband sb(nrSamples, nrChannels);
        sb.id.block = blockIdx;
        sb.id.subband = subbandIdx;

        ctr.addSubband(sb);
      }
    }

    CHECK_EQUAL(+nrBlocks, outputPool.filled.size());
    CHECK_EQUAL(0UL,       outputPool.free.size());
  }

  TEST_FIXTURE(Fixture, Loss_OneSubband) {
    // add some subbands for block 0
    for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
      // We'll drop subband 3
      if (subbandIdx == 3)
        continue;

      Subband sb(nrSamples, nrChannels);
      sb.id.block = 0;
      sb.id.subband = subbandIdx;

      ctr.addSubband(sb);
    }

    // shouldn't have emitted anything yet
    CHECK_EQUAL(0UL, outputPool.filled.size());

    // add all subbands for block 1
    for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
      Subband sb(nrSamples, nrChannels);
      sb.id.block = 1;
      sb.id.subband = subbandIdx;

      ctr.addSubband(sb);
    }

    // both blocks are now emitted, because subband 3 of block 0 is
    // considered lost by the arrival of subband 3 of block 1
    CHECK_EQUAL(2UL,            outputPool.filled.size());
    CHECK_EQUAL(nrBlocks - 2UL, outputPool.free.size());
  }

  TEST_FIXTURE(Fixture_Loss, Loss_MaxBlocksInFlight) {
    // add one subband for a new block, causing block
    // 0 to spill.
    {
      Subband sb(nrSamples, nrChannels);
      sb.id.block = maxInFlight;
      sb.id.subband = 0;

      ctr_loss.addSubband(sb);
    }

    // the first block should have been forced out
    CHECK_EQUAL(1UL,                          outputPool.filled.size());

    // some blocks are in flight, one has been emitted
    CHECK_EQUAL(nrBlocks - maxInFlight - 1UL, outputPool.free.size());

    // let subband 3 arrive late
    {
      Subband sb(nrSamples, nrChannels);
      sb.id.block = 0;
      sb.id.subband = 3;

      ctr_loss.addSubband(sb);
    }

    // there should be no change, even though this would have completed
    // a block, the block was already emitted
    CHECK_EQUAL(1UL,                          outputPool.filled.size());
    CHECK_EQUAL(nrBlocks - maxInFlight - 1UL, outputPool.free.size());
  }

  TEST_FIXTURE(Fixture, Finish) {
    // add some subbands for all blocks
    for (size_t blockIdx = 0; blockIdx < nrBlocks; ++blockIdx) {
      for (size_t subbandIdx = 0; subbandIdx < nrSubbands; ++subbandIdx) {
        // We'll drop subband 3
        if (subbandIdx == 3)
          continue;

        Subband sb(nrSamples, nrChannels);
        sb.id.block = blockIdx;
        sb.id.subband = subbandIdx;

        ctr.addSubband(sb);
      }
    }

    // shouldn't have emitted anything yet
    CHECK_EQUAL(0UL, outputPool.filled.size());
    CHECK_EQUAL(0UL, outputPool.free.size());

    ctr.finish();

    // should have emitted everything, plus
    // the terminating NULL entry
    CHECK_EQUAL(nrBlocks + 1, outputPool.filled.size());
    CHECK_EQUAL(0UL,          outputPool.free.size());
  }
}

SUITE(SendReceive) {
  TEST(NullReceiver) {
    StringStream str;
    Receiver::CollectorMap collectors;

    Receiver receiver(str, collectors);

    str.close();

    CHECK(receiver.finish());
  }

  // Test teardown if Receiver is destructed
  TEST(ReceiverDies) {
    StringStream str;
    Receiver::CollectorMap collectors;
    Receiver receiver(str, collectors);
  }

  TEST(IllegalReceive) {
    StringStream str;
    Receiver::CollectorMap collectors;

    Receiver receiver(str, collectors);

    // Receiver is not configured to receive any TABs,
    // so any write should result in the receiveLoop
    // throwing an exception.
    Subband sb;
    sb.id.fileIdx = 0;
    sb.id.block = 0;
    sb.id.subband = 0;
    sb.write(str);
    str.close();

    // Receiver thread should have thrown an exception
    CHECK(!receiver.finish());
  }

  TEST_FIXTURE(Fixture, OneToOne) {
    const size_t nrTABs = nrBlocks; // we know we have enough outputPool.free for nrBlocks TABs
    Receiver::CollectorMap collectors;

    for (size_t i = 0; i < nrTABs; ++i) {
      collectors[i] = new BlockCollector(outputPool, i);
    }

    StringStream str;

    Receiver receiver(str, collectors);

    for (size_t i = 0; i < nrTABs * nrSubbands; ++i) {
      SmartPtr<Subband> sb = new Subband(nrSamples, nrChannels);
      sb->id.fileIdx = i % nrTABs;
      sb->id.block = 0;
      sb->id.subband = i / nrTABs;

      /* fill data */
      size_t x = 0;
      for (size_t s = 0; s < nrSamples; ++s)
        for (size_t c = 0; c < nrChannels; ++c)
          sb->data[s][c] = (i+1) * ++x;

      sb->write(str);
    }

    str.close();

    // Wait for the receiver to receive and process everything
    CHECK(receiver.finish());

    // Should have one complete block
    CHECK_EQUAL(nrTABs, outputPool.filled.size());

    for (size_t i = 0; i < nrTABs; ++i) {
      SmartPtr<Block> block = outputPool.filled.remove();

      CHECK(block != NULL);
      CHECK(block->complete());

      CHECK_EQUAL(0UL, block->block);

      /* check data */
      for (size_t sb = 0; sb < nrSubbands; ++sb) {
        size_t x = 0;

        for (size_t s = 0; s < nrSamples; ++s) {
          for (size_t c = 0; c < nrChannels; ++c) {
            size_t expected = (sb * nrTABs + block->fileIdx + 1) * ++x;
            size_t actual = static_cast<size_t>(block->data[sb][s][c]);

            if (expected != actual)
              LOG_ERROR_STR("Mismatch at [" << sb << "][" << s << "][" << c << "]");
            CHECK_EQUAL(expected, actual);
          }
        }
      }
    }
  }
}

SUITE(MultiReceiver) {
  TEST(MultiReceiverDies) {
    Receiver::CollectorMap collectors;
    MultiReceiver mr("foo", collectors);
  }

  TEST(Connect) {
    Receiver::CollectorMap collectors;
    MultiReceiver mr("foo-", collectors);

    // Connect with multiple clients
    {
      PortBroker::ClientStream cs1("localhost", PortBroker::DEFAULT_PORT, "foo-1", 1);
      PortBroker::ClientStream cs2("localhost", PortBroker::DEFAULT_PORT, "foo-2", 1);

      // Disconnect them too! (~cs)
    }

    mr.kill(2);
  }

  TEST_FIXTURE(Fixture, Transfer) {
    // Set up receiver
    Receiver::CollectorMap collectors;

    collectors[0] = new BlockCollector(outputPool, 0);

    MultiReceiver mr("foo-", collectors);

    // Connect
    {
      PortBroker::ClientStream cs("localhost", PortBroker::DEFAULT_PORT, "foo-1", 1);

      // Send one block
      {
        SmartPtr<Subband> sb = new Subband(11, 13);
        sb->id.fileIdx = 0;

        sb->write(cs);
      }

      // Disconnect (~cs)
    }

    // Flush receiver, but wait for data to arrive
    mr.kill(1);

    // Flush collecting of blocks
    collectors[0]->finish();

    // Check if data arrived, resulting in one (incomplete) block,
    // plus NULL marker.
    CHECK_EQUAL(2UL, outputPool.filled.size());
  }

  TEST(MultiSender) {
    MultiSender::HostMap hostMap;
    MultiSender msender(hostMap, 3, false);
  }

  TEST(Transpose) {
    LOG_DEBUG_STR("Transpose test started");

    const size_t nrSubbands = 4;
    const size_t nrBlocks = 2;
    const size_t nrTABs = 2;
    const size_t nrSamples = 16;
    const size_t nrChannels = 1;

    // Give both senders and receivers multiple tasks,
    // but not all the same amount.
    const size_t nrSenders = 4;
    const size_t nrReceivers = 2;

    Semaphore sendersDone;

    LOG_DEBUG_STR("Spawning threads");

    // Set up sinks -- can all draw from the same outputPool
#   pragma omp parallel sections num_threads(2)
    {
      // Receivers
#     pragma omp section
      {
#       pragma omp parallel for num_threads(nrReceivers)
        for (size_t r = 0; r < nrReceivers; ++r) {
          LOG_DEBUG_STR("Receiver thread " << r);

          // Set up pool where all data ends up
          Pool<Block> outputPool;

          LOG_DEBUG_STR("Populating outputPool");

          // collect our TABs
          std::map<size_t, SmartPtr< Pool<Block> > > outputPools;
          Receiver::CollectorMap collectors;

          for (size_t t = 0; t < nrTABs; ++t) {
            if (t % nrReceivers != r)
              continue;

            outputPools[t] = new Pool<Block>;

            for (size_t i = 0; i < nrBlocks; ++i) {
              outputPools[t]->free.append(new Block(nrSubbands, nrSamples, nrChannels));
            }
            collectors[t] = new BlockCollector(*outputPools[t], t);
          }

          LOG_DEBUG_STR("Starting receiver " << r);
          MultiReceiver mr(str(format("foo-%u-") % r), collectors);

          // Wait for end of data
          LOG_DEBUG_STR("Receiver " << r << ": Waiting for senders");
          sendersDone.down();
          LOG_DEBUG_STR("Receiver " << r << ": Shutting down");
          mr.kill(nrSenders);

          // Wrap up any incomplete blocks
          for (size_t t = 0; t < nrTABs; ++t) {
            if (t % nrReceivers != r)
              continue;

            collectors[t]->finish();

            // Check if all blocks arrived, plus NULL marker.
            CHECK_EQUAL(nrBlocks + 1UL, outputPools[t]->filled.size());

            for (size_t b = 0; b < nrBlocks; ++b) {
              SmartPtr<Block> block = outputPools[t]->filled.remove();

              CHECK(block != NULL);
              CHECK(block->complete());

              // Blocks should have arrived in-order
              CHECK_EQUAL(b, block->block);
            }
          }
        }
      }

      // Senders
#     pragma omp section
      {

#       pragma omp parallel for num_threads(nrSenders)
        for (size_t s = 0; s < nrSenders; ++s) {
          LOG_DEBUG_STR("Sender thread " << s);

          MultiSender::HostMap hostMap;

          for (size_t t = 0; t < nrTABs; ++t) {
            size_t r = t % nrReceivers;

            struct MultiSender::Host host;

            host.hostName = str(format("127.0.0.%u") % (r+1));
            host.brokerPort = PortBroker::DEFAULT_PORT;
            host.service = str(format("foo-%s-%s") % r % s);

            hostMap[t] = host;
          }

          MultiSender msender(hostMap, 3, false);

#         pragma omp parallel sections num_threads(2)
          {
#           pragma omp section
            {
              msender.process();
            }

#           pragma omp section
            {
              // Send blocks
              for (size_t b = 0; b < nrBlocks; ++b) {
                // Send our subbands
                for (size_t sb = 0; sb < nrSubbands; ++sb) {
                  if (sb % nrSenders != s)
                    continue;

                  // Send all TABs
                  for (size_t t = 0; t < nrTABs; ++t) {
                    SmartPtr<Subband> subband = new Subband(nrSamples, nrChannels);
                    subband->id.fileIdx = t;
                    subband->id.block = b;
                    subband->id.subband = sb;

                    LOG_DEBUG_STR("Sender " << s << ": sending TAB " << t << " block " << b << " subband " << sb);
                    msender.append(subband);
                  }
                }
              }

              msender.finish();
            }
          }
        }

        // Signal end of data
        LOG_DEBUG_STR("Senders done.");
        sendersDone.up(nrReceivers);
      }
    }

  }
}

int main(void)
{
  INIT_LOGGER("tTABTranspose");

  omp_set_nested(true);

  PortBroker::createInstance(PortBroker::DEFAULT_PORT);

  return UnitTest::RunAllTests() > 0;
}

