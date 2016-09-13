/* ReceiveStations.h
 * Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
 * P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
 *
 * This file is part of the LOFAR software suite.
 * The LOFAR software suite is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The LOFAR software suite is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
 *
 * $Id: ReceiveStations.h 25598 2013-07-08 12:31:36Z mol $
 */

#ifndef LOFAR_INPUT_PROC_RECEIVE_STATIONS_H
#define LOFAR_INPUT_PROC_RECEIVE_STATIONS_H

#include <Common/LofarTypes.h>
#include <CoInterface/MultiDimArray.h>

#include <vector>

namespace LOFAR
{
  namespace Cobalt
  {
    /*
     * A generic Receiver class for station transposed station data, providing
     * blocks of beamlets from all specified stations.
     */
    class ReceiveStations
    {
    public:
      template<typename T>
      struct Beamlet {
        T *samples;
        SubbandMetaData metaData;
      };

      template<typename T>
      struct Block {
        std::vector< struct Beamlet<T> > beamlets; // [beamlet]
      };
    };

  }
}

#endif

