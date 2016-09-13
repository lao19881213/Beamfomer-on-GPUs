//# math.cl
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
//# $Id: math.cl 24849 2013-05-08 14:51:06Z amesfoort $

typedef float2 fcomplex;
typedef float4 fcomplex2;
typedef float8 fcomplex4;

typedef char4 char_complex2;
typedef short4 short_complex2;


fcomplex cmul(fcomplex a, fcomplex b)
{
  return (fcomplex) (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}


fcomplex cexp(float ang)
{
  return (fcomplex) (native_cos(ang), native_sin(ang));
}

