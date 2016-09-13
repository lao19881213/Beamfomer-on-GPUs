//# TriggerData.h
//# Copyright (C) 2009-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: TriggerData.h 25312 2013-06-12 15:48:13Z mol $

#ifndef LOFAR_INTERFACE_TRIGGER_DATA_H
#define LOFAR_INTERFACE_TRIGGER_DATA_H

#include <CoInterface/StreamableData.h>
#include <Stream/Stream.h>


namespace LOFAR
{
  namespace Cobalt
  {


    class TriggerData : public StreamableData
    {
    public:
      TriggerData() : trigger(false)
      {
      }

      bool trigger;

    protected:
      virtual void readData(Stream *str, unsigned)
      {
        str->read(&trigger, sizeof trigger);
      }
      virtual void writeData(Stream *str, unsigned)
      {
        str->write(&trigger, sizeof trigger);
      }
    };


  } // namespace Cobalt
} // namespace LOFAR

#endif

