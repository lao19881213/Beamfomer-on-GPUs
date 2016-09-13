//# BlockID.h
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
//# $Id: BlockID.h 26758 2013-09-30 11:47:21Z loose $

#ifndef LOFAR_GPUPROC_BLOCKID_H
#define LOFAR_GPUPROC_BLOCKID_H

#include <sys/types.h> // ssize_t
#include <iosfwd>

namespace LOFAR
{
  namespace Cobalt
  {
    struct BlockID {
      // Default constructor to provide sane defaults for our data members.
      BlockID();

      // Block number: -inf .. inf (blocks before start have negative values)
      ssize_t block;

      // Subband index in the observation: [0, ps.nrSubbands())
      size_t globalSubbandIdx;

      // Subband index for this pipeline/workqueue: [0, subbandIndices.size())
      size_t localSubbandIdx;

      // Index of this subband within the SubbandProc
      size_t subbandProcSubbandIdx;

      bool operator<(const struct BlockID &other) const;
    };

    std::ostream &operator<<(std::ostream &str, const struct BlockID &id);
  } // namespace Cobalt
} // namespace LOFAR

#endif

