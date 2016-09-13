//# global_defines.h
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
//# $Id: global_defines.h 26436 2013-09-09 16:17:42Z mol $

#ifndef LOFAR_GPUPROC_GLOBAL_DEFINES_H
#define LOFAR_GPUPROC_GLOBAL_DEFINES_H

#define NR_STATION_FILTER_TAPS  16
#define NR_POLARIZATIONS         2 // TODO: get the nr of pol symbol from an LCS/Common header and/or from CoInterface/Config.h (if that isn't a dup too)
#define NR_TAPS                 16
#undef USE_B7015

namespace LOFAR
{
  namespace Cobalt
  {
    extern bool profiling;
    extern bool gpuProfiling;

    void set_affinity(unsigned device);
  }
}

#endif

