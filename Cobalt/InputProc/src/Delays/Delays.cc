//# Delays.cc: Workholder for the delay compensation.
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
//# $Id: Delays.cc 26336 2013-09-03 10:01:41Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include "Delays.h"

#include <Common/LofarLogger.h>
#include <Common/Thread/Mutex.h>
#include <Common/Thread/Cancellation.h>
#include <CoInterface/Exceptions.h>

#ifdef HAVE_CASACORE
#include <measures/Measures/MEpoch.h>
#include <measures/Measures/MCDirection.h>
#include <casa/Exceptions/Error.h>
#endif


namespace LOFAR
{
  namespace Cobalt
  {

    //##----------------  Public methods  ----------------##//

    Delays::Delays(const Parset &parset, size_t stationIdx, const TimeStamp &from, size_t increment)
      :
      parset(parset),
      stationIdx(stationIdx),
      from(from),
      increment(increment),

      stop(false),
      // we need an extra entry for the central beam
      buffer(bufferSize, AllDelays(parset)),
      head(0),
      tail(0),
      bufferFree(bufferSize),
      bufferUsed(0),
      delayTimer("delay producer", true, true)
    {
      ASSERTSTR(test(), "Delay compensation engine is broken");
    }


    void Delays::start()
    {
      thread = new Thread(this, &Delays::mainLoop, "[DelayCompensation] ");
    }


    Delays::~Delays()
    {
      ScopedDelayCancellation dc; // Semaphores provide cancellation points

      // trigger mainLoop and force it to stop
      stop = true;
      bufferFree.up(nrCalcDelays);
    }


    void Delays::BeamDelays::read( Stream *str ) {
      size_t nrTABs;

      str->read(&SAP, sizeof SAP);

      str->read(&nrTABs, sizeof nrTABs);
      ASSERT(nrTABs == TABs.size());

      if (nrTABs > 0) {
        str->read(&TABs[0], TABs.size() * sizeof TABs[0]);
      }
    }


    void Delays::BeamDelays::write( Stream *str ) const {
      size_t nrTABs = TABs.size();

      str->write(&SAP, sizeof SAP);

      str->write(&nrTABs, sizeof nrTABs);
      if (nrTABs > 0) {
        str->write(&TABs[0], TABs.size() * sizeof TABs[0]);
      }
    }


    Delays::AllDelays::AllDelays( const Parset &parset ) {
        SAPs.resize(parset.settings.SAPs.size());

        for (size_t sap = 0; sap < parset.settings.SAPs.size(); ++sap) {
          if (parset.settings.beamFormer.enabled) {
            const struct ObservationSettings::BeamFormer::SAP &bfSap = parset.settings.beamFormer.SAPs[sap];

            SAPs[sap].TABs.resize(bfSap.TABs.size());
          }
        }
    }


    void Delays::AllDelays::read( Stream *str ) {
      size_t nrSAPs;

      str->read(&nrSAPs, sizeof nrSAPs);
      ASSERT(nrSAPs == SAPs.size());

      for (size_t n = 0; n < SAPs.size(); ++n) {
        SAPs[n].read(str);
      }
    }


    void Delays::AllDelays::write( Stream *str ) const {
      size_t nrSAPs = SAPs.size();

      str->write(&nrSAPs, sizeof nrSAPs);

      for (size_t n = 0; n < SAPs.size(); ++n) {
        SAPs[n].write(str);
      }
    }


#ifdef HAVE_CASACORE
    using namespace casa;

    static LOFAR::Mutex casacoreMutex; // casacore is not thread safe

    // convert a time in samples to a (day,fraction) pair in UTC in a CasaCore format
    MVEpoch Delays::toUTC(const TimeStamp &timestamp) const
    {
      double utc_sec = timestamp.getSeconds() / MVEpoch::secInDay;
      double day = floor(utc_sec);
      double frac = utc_sec - day;

      // (40587 modify Julian day number = 00:00:00 January 1, 1970, GMT)
      return MVEpoch(day + 40587., frac);
    }


    bool Delays::test()
    {
      try {
        ScopedLock lock(casacoreMutex);
        ScopedDelayCancellation dc;

        // set up a converter
        MDirection::Types dirType;

        if (!MDirection::getType(dirType, "J2000"))
          THROW(Exception, "Beam direction type unknown: J2000");

        MeasFrame frame;
        frame.set(MEpoch(toUTC(from), MEpoch::UTC));
        frame.set(MPosition(MVPosition(0,0,0), MPosition::ITRF));

        MDirection::Convert converter(dirType, MDirection::Ref(MDirection::ITRF, frame));

        // convert a direction
        MVDirection sourceDir(0,0);
        MVDirection pointDir = converter(sourceDir).getValue();
      } catch (AipsError &ex) {
        LOG_FATAL_STR("Casacore fails to compute delays: " << ex.what());
        LOG_FATAL_STR("Hint: If the TAI_UTC table cannot be found, please copy the measures data to ~/aips++/data, or set their location as 'measures.directory: $HOME/measures_data' or similar in ~/.casarc");
        return false;
      } catch (Exception &ex) {
        LOG_FATAL_STR("Casacore fails to compute delays: " << ex);
        return false;
      }

      return true;
    }


    void Delays::init()
    {
      ScopedLock lock(casacoreMutex);
      ScopedDelayCancellation dc;

      // We need bufferSize to be a multiple of batchSize to avoid wraparounds in
      // the middle of the batch calculations. This makes life a lot easier and there is no
      // need to support other cases.

      ASSERT(bufferSize % nrCalcDelays == 0);

      // Set an initial epoch for the frame
      frame.set(MEpoch(toUTC(from), MEpoch::UTC));

      // Set the position for the frame.
      const MVPosition phaseCenter(parset.settings.stations[stationIdx].phaseCenter);
      frame.set(MPosition(phaseCenter, MPosition::ITRF));

      // Cache the difference with CS002LBA
      const MVPosition pRef(parset.settings.delayCompensation.referencePhaseCenter);
      phasePositionDiff = phaseCenter - pRef;

      // Set-up the direction cache and conversion engines, using reference direction ITRF.
      directionTypes.resize(parset.settings.SAPs.size());

      for (size_t sap = 0; sap < parset.settings.SAPs.size(); sap++) {
        const std::string typeName = toUpper(parset.settings.SAPs[sap].direction.type);
        MDirection::Types &casaType = directionTypes[sap];

        if (!MDirection::getType(casaType, typeName))
          THROW(Exception, "Beam direction type unknown: " << typeName);

        if (converters.find(casaType) == converters.end())
          converters[casaType] = MDirection::Convert(casaType, MDirection::Ref(MDirection::ITRF, frame));
      }
    }


    struct Delays::Delay Delays::convert( casa::MDirection::Convert &converter, const casa::MVDirection &direction ) const {
      struct Delay d;

      if (parset.settings.delayCompensation.enabled) {
        MVDirection casaDir = converter(direction).getValue();

        // Compute direction and convert it 
        casa::Vector<double> dir = casaDir.getValue();
        std::copy(dir.begin(), dir.end(), d.direction);

        // Compute delay
        d.delay = casaDir * phasePositionDiff * (1.0 / speedOfLight);
      } else {
        d.delay = 0.0;
        d.direction[0] = 0.0;
        d.direction[1] = 0.0;
        d.direction[2] = 0.0;
      }

      // Add non-geometric delays
      d.delay += baseDelay();

      return d;
    }


    void Delays::calcDelays( const TimeStamp &timestamp, AllDelays &result ) {
      try {
        ScopedLock lock(casacoreMutex);
        ScopedDelayCancellation dc;

        // Set the instant in time in the frame
        frame.resetEpoch(toUTC(timestamp));

        // Convert directions for all beams
        for (size_t sap = 0; sap < parset.settings.SAPs.size(); ++sap) {
          const struct ObservationSettings::SAP &sapInfo = parset.settings.SAPs[sap];

          // Fetch the relevant convert engine
          MDirection::Convert &converter = converters[directionTypes[sap]];

          // Convert the SAP directions using the convert engine
          result.SAPs[sap].SAP = convert(converter, MVDirection(sapInfo.direction.angle1, sapInfo.direction.angle2));

          if (parset.settings.beamFormer.enabled) {
            // Convert the TAB directions using the convert engine
            const struct ObservationSettings::BeamFormer::SAP &bfSap = parset.settings.beamFormer.SAPs[sap];
            for (size_t tab = 0; tab < bfSap.TABs.size(); tab++) {
              const MVDirection dir(sapInfo.direction.angle1 + bfSap.TABs[tab].directionDelta.angle1,
                                    sapInfo.direction.angle2 + bfSap.TABs[tab].directionDelta.angle2);

              result.SAPs[sap].TABs[tab] = convert(converter, dir);
            }
          }
        }
      } catch (AipsError &ex) {
        THROW(Exception, "AipsError: " << ex.what());
      }
    }
#else
    bool Delays::test() {
      return true;
    }

    void Delays::init() {
    }

    void Delays::calcDelays( const TimeStamp &timestamp, AllDelays &result ) {
      (void)timestamp;

      for (size_t sap = 0; sap < result.SAPs.size(); ++sap) {
        result.SAPs[sap].SAP.delay = baseDelay();

        if (parset.settings.beamFormer.enabled) {
          for (size_t tab = 0; tab < result.SAPs[sap].TABs.size(); tab++) {
            result.SAPs[sap].TABs[tab].delay = baseDelay();
          }
        }
      }
    }
#endif


    double Delays::baseDelay() const
    {
      double clockCorrection = parset.settings.corrections.clock ? parset.settings.stations[stationIdx].clockCorrection : 0.0;

      return clockCorrection;
    }


    void Delays::mainLoop()
    {
      LOG_DEBUG("Delay compensation thread running");

      init();

      // the current time, in samples
      TimeStamp currentTime = from;

      try {
        while (!stop) {
          bufferFree.down(nrCalcDelays);

          delayTimer.start();

          // Calculate nrCalcDelays seconds worth of delays. Technically, we do not have
          // to calculate that many at the end of the run, but there is no need to
          // prevent the few excess delays from being calculated.

          {
            for (size_t i = 0; i < nrCalcDelays; i++) {
              // Check whether we will store results in a valid place
              ASSERTSTR(tail < bufferSize, tail << " < " << bufferSize);

              // Calculate the delays and store them in buffer[tail]
              calcDelays(currentTime, buffer[tail]);

              // Advance time for the next calculation
              currentTime += increment;

              // Advance to the next result set.
              // since bufferSize % nrCalcDelays == 0, wrap
              // around can only occur between runs
              ++tail;
            }
          }

          // check for wrap around for the next run
          if (tail >= bufferSize)
            tail = 0;

          delayTimer.stop();

          bufferUsed.up(nrCalcDelays);
        }
      } catch (Exception &ex) {
        // trigger getNextDelays and force it to stop
        stop = true;
        bufferUsed.up(1);

        throw;
      }

      LOG_DEBUG("Delay compensation thread stopped");
    }


    void Delays::getNextDelays( AllDelays &result )
    {
      ASSERT(thread);

      bufferUsed.down();

      if (stop)
        THROW(Exception, "Cannot obtain delays -- delay thread stopped running");

      // copy the directions at buffer[head]
      result = buffer[head];

      // increment the head pointer
      if (++head == bufferSize)
        head = 0;

      bufferFree.up();
    }

    void Delays::generateMetaData( const AllDelays &delaysAtBegin, const AllDelays &delaysAfterEnd, const vector<size_t> &subbands, vector<SubbandMetaData> &metaDatas, vector<ssize_t> &read_offsets )
    {
      ASSERT( metaDatas.size() == subbands.size() );
      ASSERT( read_offsets.size() == subbands.size() );

      // Delay compensation is performed in two parts. First, a coarse
      // correction is done by shifting the block to read by a whole
      // number of samples. Then, in the GPU, the remainder of the delay
      // will be corrected for using a phase shift.
      vector<ssize_t> coarseDelaysSamples(parset.settings.SAPs.size()); // [sap], in samples
      vector<double>  coarseDelaysSeconds(parset.settings.SAPs.size()); // [sap], in seconds
      for (size_t sap = 0; sap < parset.nrBeams(); ++sap) {
        double delayAtBegin  = delaysAtBegin.SAPs[sap].SAP.delay;
        double delayAfterEnd = delaysAfterEnd.SAPs[sap].SAP.delay;

        // The coarse delay compensation is based on the average delay
        // between begin and end.
        coarseDelaysSamples[sap] = static_cast<ssize_t>(floor(0.5 * (delayAtBegin + delayAfterEnd) * parset.subbandBandwidth() + 0.5));
        coarseDelaysSeconds[sap] = coarseDelaysSamples[sap] / parset.subbandBandwidth();
      }

      // Compute the offsets at which each subband is read
      for (size_t i = 0; i < subbands.size(); ++i) {
        const size_t subband = subbands[i];

        unsigned sap = parset.settings.subbands[subband].SAP;

        // Mystery unary minus
        read_offsets[i] = -coarseDelaysSamples[sap];
      }

      // Add the delays to metaDatas.
      for (size_t i = 0; i < subbands.size(); ++i) {
        const size_t subband = subbands[i];

        unsigned sap = parset.settings.subbands[subband].SAP;
        double coarseDelay = coarseDelaysSeconds[sap];

        metaDatas[i].stationBeam.delayAtBegin  = delaysAtBegin.SAPs[sap].SAP.delay - coarseDelay;
        metaDatas[i].stationBeam.delayAfterEnd = delaysAfterEnd.SAPs[sap].SAP.delay - coarseDelay;

        for (size_t tab = 0; tab < metaDatas[i].TABs.size(); ++tab) {
          metaDatas[i].TABs[tab].delayAtBegin  = delaysAtBegin.SAPs[sap].TABs[tab].delay - coarseDelay;
          metaDatas[i].TABs[tab].delayAfterEnd = delaysAfterEnd.SAPs[sap].TABs[tab].delay - coarseDelay;
        }
      }
    }

  } // namespace Cobalt
} // namespace LOFAR

