//#  RSPTimeStamp.cc: Small class to hold the timestamps from RSP
//#
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
//#  $Id: RSPTimeStamp.cc 15255 2010-03-17 13:53:24Z mol $

#include <lofar_config.h>

#include <Interface/RSPTimeStamp.h>
#include <Common/lofar_iostream.h>

namespace LOFAR {
namespace RTCP {

ostream &operator << (ostream &os, const TimeStamp &ts)
{
  return os << "[" << ts.getSeqId() << "s, " << ts.getBlockId() << "]";
}

} // namespace RTCP
} // namespace LOFAR
