//# BeamCoordinates.cc
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
//# $Id: BeamCoordinates.cc 24388 2013-03-26 11:14:29Z amesfoort $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <CoInterface/BeamCoordinates.h>

#include <Common/DataConvert.h>

#ifndef M_SQRT3
  #define M_SQRT3     1.73205080756887719000
#endif

namespace LOFAR
{
  namespace Cobalt
  {

    void BeamCoord3D::read(Stream *s)
    {
      s->read(&itsXYZ, sizeof itsXYZ);

#if !defined WORDS_BIGENDIAN
      dataConvert(LittleEndian, static_cast<double*>(itsXYZ), sizeof itsXYZ);
#endif
    }

    void BeamCoord3D::write(Stream *s) const
    {
#if !defined WORDS_BIGENDIAN
      // create a copy to avoid modifying our own values
      double coordinates[sizeof itsXYZ];

      for (unsigned i = 0; i < sizeof itsXYZ; i++)
        coordinates[i] = itsXYZ[i];

      dataConvert(LittleEndian, static_cast<double*>(coordinates), sizeof coordinates);
      s->write(&coordinates, sizeof coordinates);
#else
      s->write(&itsXYZ, sizeof itsXYZ);
#endif
    }

    void BeamCoordinates::read(Stream *s)
    {
      unsigned numCoordinates;

      s->read(&numCoordinates, sizeof numCoordinates);

#if !defined WORDS_BIGENDIAN
      dataConvert(LittleEndian, &numCoordinates, 1);
#endif

      itsCoordinates.clear();


      for (unsigned i = 0; i < numCoordinates; i++) {
        BeamCoord3D coord(0, 0, 0);

        coord.read(s);

        *this += coord;
      }
    }

    void BeamCoordinates::write(Stream *s) const
    {
      unsigned numCoordinates = itsCoordinates.size();

#if !defined WORDS_BIGENDIAN
      dataConvert(LittleEndian, &numCoordinates, 1);
#endif

      s->write(&numCoordinates, sizeof numCoordinates);

      for (unsigned i = 0; i < numCoordinates; i++)
        itsCoordinates[i].write(s);
    }

    BeamCoordinates& BeamCoordinates::operator+= (const BeamCoordinates &rhs)
    {
      itsCoordinates.reserve(itsCoordinates.size() + rhs.size());

      for (unsigned i = 0; i < rhs.size(); i++)
        itsCoordinates.push_back(rhs.itsCoordinates[i]);

      return *this;
    }

    BeamCoordinates& BeamCoordinates::operator+= (const BeamCoord3D &rhs)
    {
      itsCoordinates.push_back(rhs);

      return *this;
    }

    BeamCoordinates::BeamCoordinates(const Matrix<double> &coordinates)
    {
      itsCoordinates.reserve(coordinates.size());

      for (unsigned i = 0; i < coordinates.size(); i++)
        itsCoordinates.push_back(BeamCoord3D(coordinates[i][0], coordinates[i][1]));
    }


  }
}

