//# Exceptions.h
//# Copyright (C) 2008-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: Exceptions.h 24385 2013-03-26 10:43:55Z amesfoort $

#ifndef LOFAR_INTERFACE_EXCEPTIONS_H
#define LOFAR_INTERFACE_EXCEPTIONS_H

#include <Common/Exception.h>
#include <Common/Exceptions.h>


namespace LOFAR
{
  namespace Cobalt
  {
    //
    // This exception will be thrown when an rtcp fails.
    //
    EXCEPTION_CLASS(RTCPException, LOFAR::Exception);

    //
    // This exception will be thrown when an an CoInterface error occurs.
    //
    EXCEPTION_CLASS(CoInterfaceException, RTCPException);

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

  } // namespace Cobalt
} // namespace LOFAR

#endif

