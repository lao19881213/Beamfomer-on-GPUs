//# Delays.h: Calculate delay compensation for all stations.
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
//# $Id: Delays.h 26336 2013-09-03 10:01:41Z mol $

#ifndef LOFAR_INPUTPROC_DELAYS_H
#define LOFAR_INPUTPROC_DELAYS_H

// \file
// Calculate delay compensation for all stations.

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

//# Includes
#include <string>
#include <vector>
#include <map>

#include <Common/LofarTypes.h>
#include <Common/Timer.h>
#include <Common/Thread/Thread.h>
#include <Common/Thread/Semaphore.h>
#include <CoInterface/MultiDimArray.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SubbandMetaData.h>
#include <CoInterface/SmartPtr.h>
#include <InputProc/RSPTimeStamp.h>

#ifdef HAVE_CASACORE
#include <measures/Measures/MeasConvert.h>
#include <measures/Measures/MDirection.h>
#include <measures/Measures/MPosition.h>
#include <casa/Quanta/MVDirection.h>
#include <casa/Quanta/MVPosition.h>
#include <casa/Quanta/MVEpoch.h>
#endif

namespace LOFAR
{
  namespace Cobalt
  {

    // Speed of light in vacuum, in m/s.
    const double speedOfLight = 299792458;

    // Workholder for calculating the delay compensation that must be applied
    // per beam per station. We start by calculating the path length
    // difference for beam \f$\mathbf{b}_i\f$ between station \f$j\f$ at
    // position \f$\mathbf{p}_j\f$ and the reference station 0 at position
    // \f$\mathbf{p}_0\f$.
    // \f[
    //   d_{ij} - d_{i0} = \mathbf{b}_i \cdot \mathbf{p}_j
    //                   - \mathbf{b}_i \cdot \mathbf{p}_0
    //                   = \mathbf{b}_i \cdot (\mathbf{p}_j - \mathbf{p}_0)
    // \f]
    // The choice of reference station is arbitrary, so we simply choose the
    // first station from the parameter set. From the equation above it is
    // clear that we can reduce the number of dot products if we precalculate
    // the position difference vectors \f$\mathbf{p}_j - \mathbf{p}_0$\f,
    // which we will store in \c			itsPositionDiffs.
    //
    // The geometrical delay is easily obtained by dividing the path length
    // difference by the speed of light in vacuum. We don't need to know the
    // speed of light in the atmosphere, because the AZEL directions that
    // we've calculated are valid for vacuum (only!). This is the delay that
    // must be compensated for.
    //
    // The calculated delay compensation must be split into a coarse (whole
    // sample) delay and a fine (subsample) delay.  The coarse delay will be
    // applied in the input section as a true time delay, by shifting the
    // input samples. The fine delay will be applied in the correlator as a
    // phase shift in each frequency channel.
    class Delays
    {
    public:
      Delays(const Parset &ps, size_t stationIdx, const TimeStamp &from, size_t increment);
      ~Delays();

      void start();

      // Output structures for adjusted directions and delays
      struct Delay {
        double  direction[3];
        double  delay;

        bool operator==(const Delay &other) const {
          return direction[0] == other.direction[0] &&
                 direction[1] == other.direction[1] &&
                 direction[2] == other.direction[2] &&
                 delay == other.delay;
        }
      };

      struct BeamDelays {
        struct Delay              SAP;
        std::vector<struct Delay> TABs;

        void read( Stream *str );
        void write( Stream *str ) const;

        bool operator==(const BeamDelays &other) const {
          return SAP == other.SAP && TABs == other.TABs;
        }
      };

      class AllDelays {
      public:
        AllDelays(const Parset &ps);

        // All delays for all SAPs (and their TABs)
        std::vector<struct BeamDelays> SAPs;

        void read( Stream *str );
        void write( Stream *str ) const;

        bool operator==(const AllDelays &other) const {
          return SAPs == other.SAPs;
        }

      private:
        // Don't allow construction with illegal dimensions
        AllDelays();
      };

      // Get the set of directions (ITRF) and delays for the beams
      void getNextDelays( AllDelays &result );

      /*
       * Convert the (delaysAtBegin, delaysAfterEnd) delays pair to all delays
       * required in metaDatas, and to the read_offsets at which the data should
       * be read from the circular buffer.
       *
       * Essentially, the read_offset for each subband is 
       *     SAP.delay / sampleDuration
       * and in the metaData the remainders will be recorded, that is,
       *     SAP.delay % sampleDuration.
       */
      void generateMetaData( const AllDelays &delaysAtBegin, const AllDelays &delaysAfterEnd, const std::vector<size_t> &subbands, std::vector<SubbandMetaData> &metaDatas, std::vector<ssize_t> &read_offsets );

    private:
      const Parset &parset;
      const size_t stationIdx;
      const TimeStamp from;
      const size_t increment;

      // do the delay compensation calculations in a separate thread to allow bulk
      // calculations and to avoid blocking other threads
      void                                mainLoop();

      volatile bool stop;

      // the number of seconds to maintain in the buffer, must be a multiple of
      // nrCalcDelays.
      static const size_t bufferSize = 128;

      // the number of delays to calculate in a single run
      static const size_t nrCalcDelays = 16;

      // the circular buffer to hold the moving beam directions for every second of data
      std::vector<AllDelays> buffer;
      size_t head, tail;

      // two semaphores are used: one to trigger the producer that free space is available,
      // another to trigger the consumer that data is available.
      Semaphore bufferFree, bufferUsed;

      // Resize the given delay set to the right proportions.
      void setAllDelaysSize( AllDelays &result ) const;

      // Test whether the conversion engine actually works.
      bool test();

      // Initialise computing/converting engines
      void init();

      // Computes the delays for a specific moment in time and stores them
      // in `result'.
      void calcDelays( const TimeStamp &timestamp, AllDelays &result );

      // Returns the non-geometric delay to add for this station
      double baseDelay() const;

#ifdef HAVE_CASACORE
      casa::MVEpoch                       toUTC( const TimeStamp &timestamp ) const;

      // Converts a sky direction to a direction and delay
      struct Delay convert( casa::MDirection::Convert &converter, const casa::MVDirection &direction ) const;

      casa::MeasFrame frame;

      std::vector<casa::MDirection::Types> directionTypes; // [sap]
      std::map<casa::MDirection::Types, casa::MDirection::Convert> converters; // [type]

      // Station to reference station position difference vector.
      casa::MVPosition phasePositionDiff;
#endif

      NSTimer delayTimer;

      SmartPtr<Thread>                    thread;
    };

  } // namespace Cobalt
} // namespace LOFAR

#endif

