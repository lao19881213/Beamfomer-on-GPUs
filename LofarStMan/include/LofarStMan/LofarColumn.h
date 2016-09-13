//# LofarColumn.h: A Column in the LOFAR Storage Manager
//# Copyright (C) 2009
//# ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands
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
//# $Id: LofarColumn.h 19324 2011-11-21 07:29:55Z diepen $

#ifndef LOFAR_LOFARSTMAN_LOFARCOLUMN_H
#define LOFAR_LOFARSTMAN_LOFARCOLUMN_H


//# Includes
#include <LofarStMan/LofarStMan.h>
#include <Common/lofar_vector.h>
#include <tables/Tables/StManColumn.h>
#include <measures/Measures/MeasFrame.h>
#include <measures/Measures/MDirection.h>
#include <measures/Measures/MBaseline.h>
#include <casa/Arrays/Vector.h>
#include <casa/Arrays/IPosition.h>
#include <casa/Containers/Block.h>
#include <casa/OS/Conversion.h>

namespace LOFAR {

// <summary>
// A column in the LOFAR Storage Manager.
// </summary>

// <use visibility=local>

// <reviewed reviewer="UNKNOWN" date="before2004/08/25" tests="tLofarStMan.cc">
// </reviewed>

// <prerequisite>
//# Classes you should understand before using this one.
//   <li> <linkto class=LofarStMan>LofarStMan</linkto>
// </prerequisite>

// <synopsis>
// For each column a specific Column class exists.
// </synopsis>

class LofarColumn : public casa::StManColumn
{
public:
  explicit LofarColumn (LofarStMan* parent, int dtype)
    : StManColumn (dtype),
      itsParent   (parent)
  {}
  virtual ~LofarColumn();
  // Most columns are not writable (only DATA is writable).
  virtual casa::Bool isWritable() const;
  // Set column shape of fixed shape columns; it does nothing.
  virtual void setShapeColumn (const casa::IPosition& shape);
  // Prepare the column. By default it does nothing.
  virtual void prepareCol();
protected:
  LofarStMan* itsParent;
};

// <summary>ANTENNA1 column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class Ant1Column : public LofarColumn
{
public:
  explicit Ant1Column (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~Ant1Column();
  virtual void getIntV (casa::uInt rowNr, casa::Int* dataPtr);
};

// <summary>ANTENNA2 column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class Ant2Column : public LofarColumn
{
public:
  explicit Ant2Column (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~Ant2Column();
  virtual void getIntV (casa::uInt rowNr, casa::Int* dataPtr);
};

// <summary>TIME and TIME_CENTROID column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class TimeColumn : public LofarColumn
{
public:
  explicit TimeColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~TimeColumn();
  virtual void getdoubleV (casa::uInt rowNr, casa::Double* dataPtr);
private:
  casa::Double itsValue;
};

// <summary>INTERVAL and EXPOSURE column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class IntervalColumn : public LofarColumn
{
public:
  explicit IntervalColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~IntervalColumn();
  virtual void getdoubleV (casa::uInt rowNr, casa::Double* dataPtr);
private:
  casa::Double itsValue;
};

// <summary>All columns in the LOFAR Storage Manager with value 0.</summary>
// <use visibility=local>
class ZeroColumn : public LofarColumn
{
public:
  explicit ZeroColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~ZeroColumn();
  virtual void getIntV (casa::uInt rowNr, casa::Int* dataPtr);
private:
  casa::Int itsValue;
};

// <summary>All columns in the LOFAR Storage Manager with value False.</summary>
// <use visibility=local>
class FalseColumn : public LofarColumn
{
public:
  explicit FalseColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~FalseColumn();
  virtual void getBoolV (casa::uInt rowNr, casa::Bool* dataPtr);
private:
  casa::Bool itsValue;
};

// <summary>UVW column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class UvwColumn : public LofarColumn
{
public:
  explicit UvwColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~UvwColumn();
  virtual casa::IPosition shape (casa::uInt rownr);
  virtual void getArraydoubleV (casa::uInt rowNr,
                                casa::Array<casa::Double>* dataPtr);
  virtual void prepareCol();
private:
  casa::MDirection              itsPhaseDir;    //# could be SUN, etc.
  casa::MDirection              itsJ2000Dir;    //# Phase dir in J2000
  casa::MeasFrame               itsFrame;
  vector<casa::MBaseline>       itsAntMB;
  vector<casa::Vector<double> > itsAntUvw;
  casa::Block<bool>             itsUvwFilled;
  int                           itsLastBlNr;
  bool                          itsCanCalc;     //# false = UVW cannot be calc.
};

// <summary>DATA column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class DataColumn : public LofarColumn
{
public:
  explicit DataColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~DataColumn();
  virtual casa::Bool isWritable() const;
  virtual casa::IPosition shape (casa::uInt rownr);
  virtual void getArrayComplexV (casa::uInt rowNr,
                                 casa::Array<casa::Complex>* dataPtr);
  virtual void putArrayComplexV (casa::uInt rowNr,
                                 const casa::Array<casa::Complex>* dataPtr);
};

// <summary>FLAG column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class FlagColumn : public LofarColumn
{
public:
  explicit FlagColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~FlagColumn();
  virtual casa::IPosition shape (casa::uInt rownr);
  virtual void getArrayBoolV (casa::uInt rowNr,
                              casa::Array<casa::Bool>* dataPtr);
};

// <summary>WEIGHT column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class WeightColumn : public LofarColumn
{
public:
  explicit WeightColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~WeightColumn();
  virtual casa::IPosition shape (casa::uInt rownr);
  virtual void getArrayfloatV (casa::uInt rowNr,
                               casa::Array<casa::Float>* dataPtr);
};

// <summary>SIGMA column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class SigmaColumn : public LofarColumn
{
public:
  explicit SigmaColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~SigmaColumn();
  virtual casa::IPosition shape (casa::uInt rownr);
  virtual void getArrayfloatV (casa::uInt rowNr,
                               casa::Array<casa::Float>* dataPtr);
};

// <summary>WEIGHT_SPECTRUM column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class WSpectrumColumn : public LofarColumn
{
public:
  explicit WSpectrumColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~WSpectrumColumn();
  virtual casa::IPosition shape (casa::uInt rownr);
  virtual void getArrayfloatV (casa::uInt rowNr,
                               casa::Array<casa::Float>* dataPtr);
};

// <summary>FLAG_CATEGORY column in the LOFAR Storage Manager.</summary>
// <use visibility=local>
class FlagCatColumn : public LofarColumn
{
public:
  explicit FlagCatColumn (LofarStMan* parent, int dtype)
    : LofarColumn(parent, dtype) {}
  virtual ~FlagCatColumn();
  virtual casa::Bool isShapeDefined (casa::uInt rownr);
  virtual casa::IPosition shape (casa::uInt rownr);
};


} //# end namespace

#endif
