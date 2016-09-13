//# KernelFactory.cc
//#
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
//# $Id: KernelFactory.cc 25878 2013-08-01 02:08:38Z amesfoort $

#include <lofar_config.h>

#include "KernelFactory.h"

#include <string>
#include <boost/lexical_cast.hpp>

using namespace std;
using boost::lexical_cast;

namespace LOFAR
{
  namespace Cobalt
  {
    KernelFactoryBase::~KernelFactoryBase()
    {
    }

    CompileDefinitions
    KernelFactoryBase::compileDefinitions(const Kernel::Parameters& param) const
    {
      CompileDefinitions defs;
      // TODO: These defines are (nearly) always needed for the correlator kernels. Investigate defaults for all bf kernels. E.g. NR_STATIONS "becomes" NR_TABS after the beamformer kernel.
      defs["COMPLEX"] = "2"; // TODO: get rid of this: replace with proper complex type names
      defs["NR_CHANNELS"] = lexical_cast<string>(param.nrChannelsPerSubband);
      defs["NR_POLARIZATIONS"] = lexical_cast<string>(param.nrPolarizations);
      defs["NR_SAMPLES_PER_CHANNEL"] = 
        lexical_cast<string>(param.nrSamplesPerChannel);
      defs["NR_SAMPLES_PER_SUBBAND"] = 
        lexical_cast<string>(param.nrSamplesPerSubband);
      defs["NR_STATIONS"] = lexical_cast<string>(param.nrStations);
      return defs;
    }

    CompileFlags
    KernelFactoryBase::compileFlags(const Kernel::Parameters& /*param*/) const
    {
      CompileFlags flags;
      return flags;
    }

  }
}
