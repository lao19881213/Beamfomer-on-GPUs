//# FilterBank.h
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: FilterBank.h 25199 2013-06-05 23:46:56Z amesfoort $

#ifndef LOFAR_GPUPROC_FILTER_BANK_H
#define LOFAR_GPUPROC_FILTER_BANK_H

#define USE_ORIGINAL_FILTER 0

#include <boost/multi_array.hpp>

namespace LOFAR
{
  namespace Cobalt
  {

    enum WindowType { HAMMING, BLACKMAN, GAUSSIAN, KAISER, PREDEFINED_FILTER };

    // Note that the filter tap constants for a channel are in reverse order.
    // This makes the implementation more efficient.

    class FilterBank
    {
    public:

      // This constructor designs a new filter with the specified parameters, and initializes the weights array.
      FilterBank(bool verbose, unsigned taps, unsigned channels, WindowType windowType);

      // This constructor creates a filterbank from an already existing set of weights.
      FilterBank(bool verbose, unsigned taps, unsigned channels, float *weights);

      unsigned getNrTaps();

      float *getWeights(unsigned channel);
      const boost::multi_array<float, 2> &getWeights() const; // [nrChannels][taps];

      // In CEP, the first subband is from -98 KHz to 98 KHz, rather than from 0 to 195 KHz.
      // To avoid that the FFT outputs the channels in the wrong order (from 128 to
      // 255 followed by channels 0 to 127), we multiply each second FFT input by -1.
      // This is efficiently achieved by negating the FIR filter constants of all
      // uneven FIR filters.
      void negateWeights();

      bool isNegated();

      // Used for debugging.
      void reverseTaps();

      // Print the weights array in the natural order, in a format that can be read by gnuplot.
      void printWeights();

    private:
      // Hamming window function
      void hamming(unsigned n, double d[]);

      // Blackman window function
      void blackman(unsigned n, double d[]);

      // Gaussian window function
      void gaussian(int n, double a, double d[]);

      // Kaiser window function
      void kaiser(int n, double beta, double d[]);

      // helper functions
      double besselI0(double x);
      void interpolate(const double x[], const double y[], unsigned xlen, unsigned n, double result[]);
      void generate_fir_filter(unsigned n, double w, const double window[], double result[]);
      void generate_filter();


      // The window used for generating the filter, default is KAISER.
      WindowType itsWindowType;

      const unsigned itsNrTaps;
      const unsigned itsNrChannels;
      const bool itsVerbose;
      bool itsNegated;

      // Store the weights in a multiarray, since both the number of channels are not known at compile time.
      boost::multi_array<float, 2> weights; // [nrChannels][taps];

#if USE_ORIGINAL_FILTER
      static const float originalCepPPFWeights[256][16];
#endif

    };


    inline unsigned FilterBank::getNrTaps()
    {
      return itsNrTaps;
    }


    inline float *FilterBank::getWeights(unsigned channel)
    {
      return weights[channel].origin();
    }


    inline const boost::multi_array<float, 2> &FilterBank::getWeights() const
    {
      return weights;
    }


    inline bool FilterBank::isNegated()
    {
      return itsNegated;
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif

