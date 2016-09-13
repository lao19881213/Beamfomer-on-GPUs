//# StationInput.h: Routines to manage I/O from the stations.
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
//# $Id: StationInput.h 27103 2013-10-27 09:51:53Z mol $


#ifndef LOFAR_GPUPROC_STATIONINPUT_H
#define LOFAR_GPUPROC_STATIONINPUT_H

#include <map>
#include <vector>

#include <Common/Thread/Semaphore.h>
#include <Common/Singleton.h>
#include <InputProc/Buffer/BlockReader.h>
#include <InputProc/Transpose/ReceiveStations.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SubbandMetaData.h>
#include <CoInterface/BestEffortQueue.h>

namespace LOFAR {
  namespace Cobalt {

#ifndef HAVE_MPI
    class DirectInput: public ReceiveStations {
    public:
      // The first call should provide the parset to allow
      // the instance to be constructed.
      static DirectInput &instance(const Parset *ps = NULL);

      template<typename T> void sendBlock(unsigned stationIdx, const struct BlockReader<T>::LockedBlock &block, const vector<SubbandMetaData> &metaDatas);

      template<typename T> void receiveBlock(std::vector<struct ReceiveStations::Block<T> > &block);

    private:
      DirectInput(const Parset &ps);

      const Parset ps;

      struct InputBlock {
        std::vector<char> samples;
        SubbandMetaData metaData;
      };

      MultiDimArray< SmartPtr< BestEffortQueue< SmartPtr<struct InputBlock> > >, 2> stationDataQueues; // [stationIdx][globalSubbandIdx]
    };
#endif

    // Which MPI rank receives which subbands?
    typedef std::map<int, std::vector<size_t> > SubbandDistribution;

    void sendInputToPipeline(const Parset &ps, size_t stationIdx, const SubbandDistribution &subbandDistribution);
  }
}

#endif

