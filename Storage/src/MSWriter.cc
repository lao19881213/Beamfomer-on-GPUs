//  MSMriter.cc: Base classs for MS writer
//
//  Copyright (C) 2008
//  ASTRON (Netherlands Foundation for Research in Astronomy)
//  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  $Id: MSWriter.cc 22857 2012-11-20 08:58:52Z mol $
//
//////////////////////////////////////////////////////////////////////

#include <lofar_config.h>

#include <Storage/MSWriter.h>
#include <algorithm>


namespace LOFAR {
namespace RTCP {

MSWriter::MSWriter()
:
  itsNrBlocksWritten(0),
  itsNrExpectedBlocks(0)
{
}


MSWriter::~MSWriter()
{
}

void MSWriter::augment(const FinalMetaData &finalMetaData)
{
  (void)finalMetaData;
}


size_t MSWriter::getDataSize()
{
  return 0;
}

ParameterSet MSWriter::configuration() const
{
  return itsConfiguration;
}


/* Returns a percentage based on a current and a target value,
 * with the following rounding:
 *
 * 0     -> current == 0
 * 1..99 -> 0 < current < target
 * 100   -> current == target
 */

unsigned MSWriter::percentageWritten() const
{
  size_t current = itsNrBlocksWritten;
  size_t target = itsNrExpectedBlocks;

  if (current == target || target == 0)
    return 100;

  if (current == 0)
    return 0;

  return std::min(std::max(100 * current / target, static_cast<size_t>(1)), static_cast<size_t>(99));
}


}
}
