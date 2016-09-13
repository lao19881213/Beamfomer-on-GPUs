//#  Exceptions.h:
//#
//#  Copyright (C) 2007
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
//#  $Id: Exceptions.h 23193 2012-12-06 15:07:04Z mol $

#ifndef LOFAR_INTERFACE_EXCEPTIONS_H
#define LOFAR_INTERFACE_EXCEPTIONS_H

#include <Common/Exception.h>
#include <Common/Exceptions.h>


namespace LOFAR {
namespace RTCP {
  //
  // This exception will be thrown when an rtcp fails.
  //
  EXCEPTION_CLASS(RTCPException, LOFAR::Exception);

  //
  // This exception will be thrown when an an Interface error occurs.
  //
  EXCEPTION_CLASS(InterfaceException, RTCPException);

  //
  // This exception will be thrown when an an IONProc error occurs.
  //
  EXCEPTION_CLASS(IONProcException, RTCPException);

  //
  // This exception will be thrown when an an CNProc error occurs.
  //
  EXCEPTION_CLASS(CNProcException, RTCPException);

  //
  // This exception will be thrown when an an GPUProc error occurs.
  //
  EXCEPTION_CLASS(GPUProcException, RTCPException);

  //
  // This exception will be thrown when an an Storage error occurs.
  //
  EXCEPTION_CLASS(StorageException, RTCPException);

} // namespace RTCP
} // namespace LOFAR

#endif 
