//# RSPTimeStamp.cc: Small class to hold the timestamps from RSP
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
//# $Id: RSPTimeStamp.cc 25497 2013-06-27 05:36:14Z mol $

#include <lofar_config.h>

#include "RSPTimeStamp.h"

#include <Common/lofar_iostream.h>
#include <Common/lofar_iomanip.h>

#include <time.h>

namespace LOFAR
{
  namespace Cobalt
  {

    ostream &operator << (ostream &os, const TimeStamp &ts)
    {
      double time_d = ts.getSeconds();
      time_t seconds = static_cast<time_t>(floor(time_d));
      unsigned ms = static_cast<unsigned>(floor((time_d - seconds) * 1000 + 0.5));

      char   buf[26];
      struct tm tm;

      gmtime_r(&seconds, &tm);
      size_t len = strftime(buf, sizeof buf, "%F %T", &tm);
      buf[len] = '\0';

      return os << "[" << ts.getSeqId() << "s, " << ts.getBlockId() << "] = " << buf << "." << setfill('0') << setw(3) << ms << " UTC";
    }

  } // namespace Cobalt
} // namespace LOFAR

