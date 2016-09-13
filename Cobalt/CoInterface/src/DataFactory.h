//# DataFactory.h
//# Copyright (C) 2011-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: DataFactory.h 25662 2013-07-13 19:10:59Z mol $

#ifndef LOFAR_INTERFACE_DATA_FACTORY_H
#define LOFAR_INTERFACE_DATA_FACTORY_H

#include <CoInterface/Allocator.h>
#include <CoInterface/OutputTypes.h>
#include <CoInterface/Parset.h>
#include <CoInterface/StreamableData.h>


namespace LOFAR
{
  namespace Cobalt
  {

    StreamableData *newStreamableData(const Parset &, OutputType, unsigned streamNr, Allocator & = heapAllocator);

  } // namespace Cobalt
} // namespace LOFAR

#endif

