//#  FCNP_ClientStream.h: Stream that implements FCNP protocol
//#
//#  Copyright (C) 2005
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
//#  $Id: FCNP_ClientStream.h 15458 2010-04-15 15:32:36Z romein $

#ifndef LOFAR_CNPROC_FCNP_CLIENT_STREAM_H
#define LOFAR_CNPROC_FCNP_CLIENT_STREAM_H

#if defined HAVE_FCNP && defined HAVE_BGP_CN

#include <Stream/Stream.h>

namespace LOFAR {
namespace RTCP {  

class FCNP_ClientStream : public Stream
{
  public:
		   FCNP_ClientStream(unsigned channel);
    virtual	   ~FCNP_ClientStream();

    virtual size_t tryRead(void *ptr, size_t size);
    virtual size_t tryWrite(const void *ptr, size_t size);

  private:
    const unsigned itsChannel;
};


inline FCNP_ClientStream::FCNP_ClientStream(unsigned channel)
:
  itsChannel(channel)
{
}

} // namespace RTCP
} // namespace LOFAR

#endif
#endif
