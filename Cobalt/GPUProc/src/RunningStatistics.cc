//# RunningStatistics.cc
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
//# $Id: RunningStatistics.cc 26143 2013-08-21 12:36:16Z klijn $
#include "RunningStatistics.h"
#include <math.h> 
#include <limits>
#include <ostream>
#include <iomanip>
namespace LOFAR
{
  namespace Cobalt
  {

    RunningStatistics::RunningStatistics() 
    {
      reset();
    }

    void RunningStatistics::reset()
    {
      counter = 0;
      _mean = var_base =  0.0;
      _min = std::numeric_limits<double>::max();
      _max = std::numeric_limits<double>::min();
    }

    void RunningStatistics::push(double x)
    {
      // update the maxima
      if (x < _min)
        _min = x;
      if (x > _max)
        _max = x;

      double delta_mean, delta_weighted;
      long long old_counter = counter;

      counter++;
      delta_mean = x - _mean;
      delta_weighted = delta_mean / counter;
      _mean += delta_weighted;
      var_base += delta_mean * delta_weighted * old_counter;
    }

    size_t RunningStatistics::count() const
    {
      return counter;
    }

    double RunningStatistics::mean() const
    {
      return _mean;
    }

    double RunningStatistics::variance() const
    {
      return  counter > 1? var_base/(counter -1): 0.0;
    }

    double RunningStatistics::stDev() const
    {
      return sqrt(variance());
    }

          // Minimum value received samples
    double RunningStatistics::min() const
    {
      return _min;
    }

    // Maximum value received samples
    double RunningStatistics::max() const
    {
      return _max;
    }

    RunningStatistics& RunningStatistics::operator+=(const RunningStatistics& other)
    { 
      RunningStatistics combined = *this + other;
      *this = combined;

      return *this;
    }

    RunningStatistics operator+(const RunningStatistics left,
         const RunningStatistics right)
    {
      RunningStatistics combined;
      combined.counter = left.counter + right.counter;

      // If both counters are empty return empty counter
      if (combined.counter == 0)
        return combined;

      combined._min = left._min < right._min ? left._min : right._min;
      combined._max = left._max > right._max ? left._max : right._max;

      double delta_means = right._mean - left._mean;
      double delta_means_sqrt = delta_means*delta_means;

      // get the weighted mean of the two inputs
      combined._mean = (left.counter * left._mean + right.counter * right._mean) / combined.counter;


      combined.var_base = left.var_base + right.var_base + 
        delta_means_sqrt * left.counter * right.counter / combined.counter;

      return combined;
    }

    void RunningStatistics::print(std::ostream& os) const
    {
      if (count() == 0)      
        os << "*Not executed*";
      else
      {
        os.precision(5);

        os  << " count: "  << std::setw(8) << count()
            << " mean:  "  << std::setw(8) << mean() 
            << " stDev: " << std::setw(8) << stDev()
            << " min:   " << std::setw(8) << min()    
            << " max:   " << std::setw(8) << max()
            << " (stats in ms)" ;
      }
    }

    std::ostream& operator<<(std::ostream& os, RunningStatistics const & rs)
    {
      rs.print(os);
      return os;
    }
  }
}