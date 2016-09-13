//#  Stream.h: functions to construct streams between ION/CN/Storage
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: Stream.h 16488 2010-10-07 10:06:14Z mol $

#ifndef LOFAR_INTERFACE_STREAM_H
#define LOFAR_INTERFACE_STREAM_H

#include <Interface/OutputTypes.h>
#include <Interface/Parset.h>
#include <Stream/Stream.h>

#include <string>

namespace LOFAR {
namespace RTCP {

// Create a stream from a descriptor
Stream *createStream(const std::string &descriptor, bool asReader, time_t deadline = 0);

// Return a string descriptor, for all supported streamTypes except FCNP
std::string getStreamDescriptorBetweenIONandCN(const char *streamType, unsigned ionode, unsigned pset, unsigned core, unsigned numpsets, unsigned numcores, unsigned channel);

uint16 storageBrokerPort(int observationID);
std::string getStorageControlDescription(int observationID, int rank);

std::string getStreamDescriptorBetweenIONandStorage(const Parset &parset, OutputType outputType, unsigned streamNr);

} // namespace RTCP
} // namespace LOFAR

#endif
