//# MPI_RMA.h
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
//# $Id: MPI_RMA.h 25015 2013-05-22 22:35:17Z amesfoort $

#ifndef LOFAR_INPUT_PROC_MPI_RMA_H
#define LOFAR_INPUT_PROC_MPI_RMA_H

#define MULTIPLE_WINDOWS

template<typename T>
class MPISharedBuffer : public SampleBuffer<T>
{
public:
  MPISharedBuffer( const struct BufferSettings &settings );

  ~MPISharedBuffer();

private:
#ifdef MULTIPLE_WINDOWS
  std::vector<MPI_Win> beamlets_windows;
#else
  MPI_Win beamlets_window;
#endif
};

template<typename T>
MPISharedBuffer<T>::MPISharedBuffer( const struct BufferSettings &settings )
  :
  SampleBuffer<T>(settings, false)
#ifdef MULTIPLE_WINDOWS
  , beamlets_windows(NRSTATIONS)
#endif
{
#ifdef MULTIPLE_WINDOWS
  for (int i = 0; i < NRSTATIONS; ++i) {
    int error = MPI_Win_create(this->beamlets.origin(), this->beamlets.num_elements() * sizeof(T), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &beamlets_windows[i]);

    ASSERT(error == MPI_SUCCESS);
  }
#else
  int error = MPI_Win_create(this->beamlets.origin(), this->beamlets.num_elements() * sizeof(T), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &beamlets_window);

  ASSERT(error == MPI_SUCCESS);
#endif
}

template<typename T>
MPISharedBuffer<T>::~MPISharedBuffer()
{
#ifdef MULTIPLE_WINDOWS
  for (int i = 0; i < NRSTATIONS; ++i) {
    int error = MPI_Win_free(&beamlets_windows[i]);

    ASSERT(error == MPI_SUCCESS);
  }
#else
  int error = MPI_Win_free(&beamlets_window);

  ASSERT(error == MPI_SUCCESS);
#endif
}

template<typename T>
class MPISharedBufferReader
{
public:
  MPISharedBufferReader( const std::vector<struct BufferSettings> &settings, const TimeStamp &from, const TimeStamp &to, size_t blockSize, const std::vector<size_t> &beamlets );

  ~MPISharedBufferReader();

  void process( double maxDelay );

private:
  const std::vector<struct BufferSettings> settings;
  const TimeStamp from, to;
  const size_t blockSize;
  const std::vector<size_t> beamlets;

  MultiDimArray<T, 3> buffer; // [station][beamlet][sample]

#ifdef MULTIPLE_WINDOWS
  std::vector<MPI_Win> beamlets_windows;
#else
  MPI_Win beamlets_window;
#endif

  WallClockTime waiter;

  void copy( const TimeStamp &from, const TimeStamp &to );
};

template<typename T>
MPISharedBufferReader<T>::MPISharedBufferReader( const std::vector<struct BufferSettings> &settings, const TimeStamp &from, const TimeStamp &to, size_t blockSize, const std::vector<size_t> &beamlets )
  :
  settings(settings),
  from(from),
  to(to),
  blockSize(blockSize),
  beamlets(beamlets),

  buffer(boost::extents[settings.size()][beamlets.size()][blockSize], 128, heapAllocator, false, false)
#ifdef MULTIPLE_WINDOWS
  , beamlets_windows(settings.size())
#endif
{
  ASSERT( settings.size() > 0 );
  ASSERT( from.getClock() == to.getClock() );
  ASSERT( settings[0].station.clock == from.getClock());

  for (size_t i = 0; i < settings.size(); ++i) {
    ASSERT(settings[i].station.clock   == settings[0].station.clock);
    ASSERT(settings[i].station.clock   == from.getClock());
    ASSERT(settings[i].station.bitmode == settings[0].station.bitmode);

    ASSERT(settings[i].nrSamples > blockSize);
  }

#ifdef MULTIPLE_WINDOWS
  for (int i = 0; i < settings.size(); ++i) {
    int error = MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &beamlets_windows[i]);

    ASSERT(error == MPI_SUCCESS);
  }
#else
  int error = MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &beamlets_window);

  ASSERT(error == MPI_SUCCESS);
#endif
}

template<typename T>
MPISharedBufferReader<T>::~MPISharedBufferReader()
{
#ifdef MULTIPLE_WINDOWS
  for (int i = 0; i < settings.size(); ++i) {
    int error = MPI_Win_free(&beamlets_windows[i]);

    ASSERT(error == MPI_SUCCESS);
  }
#else
  int error = MPI_Win_free(&beamlets_window);

  ASSERT(error == MPI_SUCCESS);
#endif
}

template<typename T>
void MPISharedBufferReader<T>::process( double maxDelay )
{
  const TimeStamp maxDelay_ts(static_cast<int64>(maxDelay * settings[0].station.clock / 1024) + blockSize, settings[0].station.clock);

  const TimeStamp current(from);

  double totalwait = 0.0;
  unsigned totalnr = 0;

  double lastreport = MPI_Wtime();

  for (TimeStamp current = from; current < to; current += blockSize) {
    // wait
    //LOG_INFO_STR("Waiting until " << (current + maxDelay_ts) << " for " << current);
    waiter.waitUntil( current + maxDelay_ts );

    // read
    //LOG_INFO_STR("Reading from " << current << " to " << (current + blockSize));
    double bs = MPI_Wtime();

    copy(current, current + blockSize);

    totalwait += MPI_Wtime() - bs;
    totalnr++;

    if (bs - lastreport > 1.0) {
      double mbps = (sizeof(T) * blockSize * beamlets.size() * 8) / (totalwait / totalnr) / 1e6;
      lastreport = bs;
      totalwait = 0.0;
      totalnr = 0;

      LOG_INFO_STR("Reading speed: " << mbps << " Mbit/s");
    }
  }

  LOG_INFO("Done reading data");
}

template<typename T>
void MPISharedBufferReader<T>::copy( const TimeStamp &from, const TimeStamp &to )
{
  int error;

#ifdef MULTIPLE_WINDOWS
  for (int i = 0; i < settings.size(); ++i) {
    error = MPI_Win_lock( MPI_LOCK_SHARED, i, MPI_MODE_NOCHECK, beamlets_windows[i] );
    ASSERT(error == MPI_SUCCESS);
  }
#endif

  for (size_t s = 0; s < settings.size(); ++s) {
#ifndef MULTIPLE_WINDOWS
    error = MPI_Win_lock( MPI_LOCK_SHARED, s, MPI_MODE_NOCHECK, beamlets_window );
    ASSERT(error == MPI_SUCCESS);
#endif

    //LOG_INFO_STR("Copying from station " << s);
    const struct BufferSettings settings = this->settings[s];

    size_t from_offset = (int64)from % settings.nrSamples;
    size_t to_offset = (int64)to % settings.nrSamples;

    if (to_offset == 0)
      to_offset = settings.nrSamples;

    size_t wrap = from_offset < to_offset ? 0 : settings.nrSamples - from_offset;

    for (size_t i = 0; i < beamlets.size(); ++i) {
      unsigned nr = beamlets[i];

      size_t origin = nr * settings.nrSamples;

      if (wrap > 0) {
        //if (i==0) LOG_INFO_STR("Reading wrapped data");
#ifdef MULTIPLE_WINDOWS
        error = MPI_Get( &buffer[s][i][0], wrap * sizeof(T), MPI_CHAR, s, (origin + from_offset) * sizeof(T), wrap * sizeof(T), MPI_CHAR, beamlets_windows[s] );
#else
        error = MPI_Get( &buffer[s][i][0], wrap * sizeof(T), MPI_CHAR, s, (origin + from_offset) * sizeof(T), wrap * sizeof(T), MPI_CHAR, beamlets_window );
#endif

        ASSERT(error == MPI_SUCCESS);

#ifdef MULTIPLE_WINDOWS
        error = MPI_Get( &buffer[s][i][wrap], to_offset * sizeof(T), MPI_CHAR, s, origin * sizeof(T), to_offset * sizeof(T), MPI_CHAR, beamlets_windows[s] );
#else
        error = MPI_Get( &buffer[s][i][wrap], to_offset * sizeof(T), MPI_CHAR, s, origin * sizeof(T), to_offset * sizeof(T), MPI_CHAR, beamlets_window );
#endif

        ASSERT(error == MPI_SUCCESS);
      } else {
        // higher performance by splitting into multiple requests if block size is large -- formula yet unknown
        //size_t partSize = (to_offset - from_offset) / 2 + 1;
        size_t partSize = to_offset - from_offset;

        for (size_t x = from_offset; x < to_offset; x += partSize) {
          size_t y = std::min(x + partSize, to_offset);

#ifdef MULTIPLE_WINDOWS
          error = MPI_Get( &buffer[s][i][x - from_offset], (y - x) * sizeof(T), MPI_CHAR, s, (origin + x) * sizeof(T), (y - x) * sizeof(T), MPI_CHAR, beamlets_windows[s] );
#else
          error = MPI_Get( &buffer[s][i][x - from_offset], (y - x) * sizeof(T), MPI_CHAR, s, (origin + x) * sizeof(T), (y - x) * sizeof(T), MPI_CHAR, beamlets_window );
#endif

          ASSERT(error == MPI_SUCCESS);
        }
      }
    }

#ifndef MULTIPLE_WINDOWS
    error = MPI_Win_unlock( s, beamlets_window );
    ASSERT(error == MPI_SUCCESS);
#endif
  }

#ifdef MULTIPLE_WINDOWS
  for (int i = 0; i < settings.size(); ++i) {
    error = MPI_Win_unlock( i, beamlets_windows[i] );
    ASSERT(error == MPI_SUCCESS);
  }
#endif
}

#endif

