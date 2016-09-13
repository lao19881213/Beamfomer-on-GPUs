//# RunningStatistics.h
//#
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: RunningStatistics.h 26143 2013-08-21 12:36:16Z klijn $

#ifndef LOFAR_GPUPROC_RUNNINGSTATISTICS_H
#define LOFAR_GPUPROC_RUNNINGSTATISTICS_H

#include <cstddef>
#include <iosfwd>

namespace LOFAR
{
  namespace Cobalt
  {
    // RunningStatistics collects mean and variance for a running
    // process without storing the individual entries.
    // Donald Knuth's Art of Computer Programming, Vol 2, page 232, 3rd edition. 
    // http://www.johndcook.com/standard_deviation.html
    class RunningStatistics
    {
    public:
      // Constructor
      RunningStatistics(); 
        
      // Reset to zero
      void reset();

      // insert a new value
      void push(double sample);
      
      // number of inserted samples
      size_t count() const;

      // mean of the received samples
      double mean() const;

      // The variance of the received samples
      double variance() const;

      // The standard deviation
      double stDev() const;

      // Minimum value received samples
      double min() const;

      // Maximum value received samples
      double max() const;

           
      friend RunningStatistics operator+(const RunningStatistics a,
        const RunningStatistics b);

      // Plus is operator for adding two runningstatistics objects
      RunningStatistics& operator+=(const RunningStatistics& rhs);

      void print(std::ostream& os) const;

    private:
      size_t counter;
      double _mean;
      double var_base;
      double _min;
      double _max;
    };

    RunningStatistics operator+(const RunningStatistics a, const RunningStatistics b);    

    std::ostream& operator<<(std::ostream& os, RunningStatistics const & rs);
  }
}
#endif
