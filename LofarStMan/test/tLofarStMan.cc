//# tLofarStMan.cc: Test program for the LofarStMan class
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
//# $Id: tLofarStMan.cc 20942 2012-05-15 13:43:52Z diepen $

#include <LofarStMan/LofarStMan.h>
#include <tables/Tables/TableDesc.h>
#include <tables/Tables/SetupNewTab.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableLock.h>
#include <tables/Tables/ScaColDesc.h>
#include <tables/Tables/ArrColDesc.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/ArrayColumn.h>
#include <measures/TableMeasures/TableMeasRefDesc.h>
#include <measures/TableMeasures/TableMeasValueDesc.h>
#include <measures/TableMeasures/TableMeasDesc.h>
#include <measures/TableMeasures/ScalarMeasColumn.h>
#include <measures/TableMeasures/ArrayMeasColumn.h>
#include <measures/Measures/MDirection.h>
#include <measures/Measures/MCDirection.h>
#include <measures/Measures/MPosition.h>
#include <measures/Measures/MCPosition.h>
#include <casa/Arrays/Array.h>
#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/ArrayIO.h>
#include <casa/Arrays/ArrayLogical.h>
#include <casa/Containers/BlockIO.h>
#include <casa/OS/RegularFile.h>
#include <casa/Utilities/Assert.h>
#include <casa/IO/RegularFileIO.h>
#include <casa/IO/RawIO.h>
#include <casa/IO/CanonicalIO.h>
#include <casa/OS/HostInfo.h>
#include <casa/Exceptions/Error.h>
#include <casa/iostream.h>
#include <casa/sstream.h>

using namespace LOFAR;
using namespace casa;

// This program tests the class LofarStMan and related classes.
// The results are written to stdout. The script executing this program,
// compares the results with the reference output file.
//
// It uses the given phase center, station positions, times, and baselines to
// get known UVW coordinates. In this way the UVW calculation can be checked.


void createField (Table& mainTable)
{
  // Build the table description.
  TableDesc td;
  td.addColumn (ArrayColumnDesc<Double>("PHASE_DIR"));
  TableMeasRefDesc measRef(MDirection::J2000);
  TableMeasValueDesc measVal(td, "PHASE_DIR");
  TableMeasDesc<MDirection> measCol(measVal, measRef);
  measCol.write (td);
  // Now create a new table from the description.
  SetupNewTable newtab("tLofarStMan_tmp.data/FIELD", td, Table::New);
  Table tab(newtab, 1);
  ArrayMeasColumn<MDirection> fldcol (tab, "PHASE_DIR");
  Vector<MDirection> phaseDir(1);
  phaseDir[0] = MDirection(MVDirection(Quantity(-1.92653768, "rad"),
                                       Quantity(1.09220917, "rad")),
                           MDirection::J2000);
  fldcol.put (0, phaseDir);
  mainTable.rwKeywordSet().defineTable ("FIELD", tab);
}

void createAntenna (Table& mainTable, uInt nant)
{
  // Build the table description.
  TableDesc td;
  td.addColumn (ArrayColumnDesc<Double>("POSITION"));
  TableMeasRefDesc measRef(MPosition::ITRF);
  TableMeasValueDesc measVal(td, "POSITION");
  TableMeasDesc<MPosition> measCol(measVal, measRef);
  measCol.write (td);
  // Now create a new table from the description.
  SetupNewTable newtab("tLofarStMan_tmp.data/ANTENNA", td, Table::New);
  Table tab(newtab, nant);
  ScalarMeasColumn<MPosition> poscol (tab, "POSITION");
  // Use the positions of WSRT RT0-3.
  Vector<double> vals(3);
  vals[0] = 3828763; vals[1] = 442449; vals[2] = 5064923;
  poscol.put (0, MPosition(Quantum<Vector<double> >(vals,"m"),
                           MPosition::ITRF));
  vals[0] = 3828746; vals[1] = 442592; vals[2] = 5064924;
  poscol.put (1, MPosition(Quantum<Vector<double> >(vals,"m"),
                           MPosition::ITRF));
  vals[0] = 3828729; vals[1] = 442735; vals[2] = 5064925;
  poscol.put (2, MPosition(Quantum<Vector<double> >(vals,"m"),
                           MPosition::ITRF));
  vals[0] = 3828713; vals[1] = 442878; vals[2] = 5064926;
  poscol.put (3, MPosition(Quantum<Vector<double> >(vals,"m"),
                           MPosition::ITRF));
  // Write the remaining columns with the same position.
  // They are not used in the UVW check.
  for (uInt i=4; i<nant; ++i) {
    poscol.put (i, MPosition(Quantum<Vector<double> >(vals,"m"),
                             MPosition::ITRF));
  }
  mainTable.rwKeywordSet().defineTable ("ANTENNA", tab);
}

void createTable (uInt nant)
{
  // Build the table description.
  // Add all mandatory columns of the MS main table.
  TableDesc td("", "1", TableDesc::Scratch);
  td.comment() = "A test of class Table";
  td.addColumn (ScalarColumnDesc<Double>("TIME"));
  td.addColumn (ScalarColumnDesc<Int>("ANTENNA1"));
  td.addColumn (ScalarColumnDesc<Int>("ANTENNA2"));
  td.addColumn (ScalarColumnDesc<Int>("FEED1"));
  td.addColumn (ScalarColumnDesc<Int>("FEED2"));
  td.addColumn (ScalarColumnDesc<Int>("DATA_DESC_ID"));
  td.addColumn (ScalarColumnDesc<Int>("PROCESSOR_ID"));
  td.addColumn (ScalarColumnDesc<Int>("FIELD_ID"));
  td.addColumn (ScalarColumnDesc<Int>("ARRAY_ID"));
  td.addColumn (ScalarColumnDesc<Int>("OBSERVATION_ID"));
  td.addColumn (ScalarColumnDesc<Int>("STATE_ID"));
  td.addColumn (ScalarColumnDesc<Int>("SCAN_NUMBER"));
  td.addColumn (ScalarColumnDesc<Double>("INTERVAL"));
  td.addColumn (ScalarColumnDesc<Double>("EXPOSURE"));
  td.addColumn (ScalarColumnDesc<Double>("TIME_CENTROID"));
  td.addColumn (ScalarColumnDesc<Bool>("FLAG_ROW"));
  td.addColumn (ArrayColumnDesc<Double>("UVW",IPosition(1,3),
                                        ColumnDesc::Direct));
  td.addColumn (ArrayColumnDesc<Complex>("DATA"));
  td.addColumn (ArrayColumnDesc<Float>("SIGMA"));
  td.addColumn (ArrayColumnDesc<Float>("WEIGHT"));
  td.addColumn (ArrayColumnDesc<Float>("WEIGHT_SPECTRUM"));
  td.addColumn (ArrayColumnDesc<Bool>("FLAG"));
  td.addColumn (ArrayColumnDesc<Bool>("FLAG_CATEGORY"));
  // Now create a new table from the description.
  SetupNewTable newtab("tLofarStMan_tmp.data", td, Table::New);
  // Create the storage manager and bind all columns to it.
  LofarStMan sm1;
  newtab.bindAll (sm1);
  // Finally create the table. The destructor writes it.
  Table tab(newtab);
  // Create the subtables needed to calculate UVW coordinates.
  createField (tab);
  createAntenna (tab, nant);
}

uInt nalign (uInt size, uInt alignment)
{
  return (size + alignment-1) / alignment * alignment - size;
}

void createData (uInt nseq, uInt nant, uInt nchan, uInt npol,
                 Double startTime, Double interval, const Complex& startValue,
                 uInt alignment, Bool bigEndian, uInt myStManVersion,
		 uInt myNrBytesPerValidSamples=2)
{
  // Create the baseline vectors (no autocorrelations).
  uInt nrbl = nant*nant;
  Block<Int> ant1(nrbl);
  Block<Int> ant2(nrbl);
  uInt inx=0;

  for (uInt i=0; i<nant; ++i) {
    for (uInt j=0; j<nant; ++j) {
      if (myStManVersion != 2) {
        // Use baselines 0,0, 0,1, 1,1 ... n,n
	ant1[inx] = j;
	ant2[inx] = i;
      } else {
        // Use baselines 0,0, 1,0, 1,1 ... n,n (will be swapped by LofarStMan)
	ant1[inx] = i;
	ant2[inx] = j;
      } 
      ++inx;
    }
  }
  // Create the meta file. The destructor closes it.
  {
    // If not big-endian, use local format (which might be big-endian).
    if (!bigEndian) {
      bigEndian = HostInfo::bigEndian();
    }
    double maxNSample = 32768;

    AlwaysAssertExit(myStManVersion <= 3);

    AipsIO aio("tLofarStMan_tmp.data/table.f0meta", ByteIO::New);
    aio.putstart ("LofarStMan", myStManVersion);     // version 1, 2, or 3
    aio << ant1 << ant2 << startTime << interval << nchan
        << npol << maxNSample << alignment << bigEndian;
    if (myStManVersion > 1) {
      aio << myNrBytesPerValidSamples;
    }
    aio.close();
  }
  // Now create the data file.
  RegularFileIO file(RegularFile("tLofarStMan_tmp.data/table.f0data"),
                     ByteIO::New);
  // Write in canonical (big endian) or local format.
  TypeIO* cfile;
  if (bigEndian) {
    cfile = new CanonicalIO(&file);
  } else {
    cfile = new RawIO(&file);
  }

  // Create and initialize data and nsample.
  Array<Complex> data(IPosition(2,npol,nchan));
  indgen (data, startValue, Complex(0.01, 0.01));

  Array<uChar>  nsample1(IPosition(1, nchan));
  Array<uShort> nsample2(IPosition(1, nchan));
  Array<uInt>   nsample4(IPosition(1, nchan));

  indgen (nsample1);
  indgen (nsample2);
  indgen (nsample4);


  // Allocate space for possible block alignment.
  if (alignment < 1) {
    alignment = 1;
  }
  Block<Char> align1(nalign(4, alignment), 0);
  Block<Char> align2(nalign(nrbl*8*data.size(), alignment), 0);

  uInt nsamplesSize=0;
  if (myStManVersion < 2) {
    nsamplesSize = nrbl*2*nsample2.size();
  } else {
    nsamplesSize = nrbl*myNrBytesPerValidSamples*nsample2.size();
  }

  Block<Char> align3(nalign(nsamplesSize, alignment), 0);

  // Write the data as nseq blocks.
  for (uInt i=0; i<nseq; ++i) {
    cfile->write (1, &i);

    if (align1.size() > 0) {
      cfile->write (align1.size(), align1.storage());
    }
    for (uInt j=0; j<nrbl; ++j) {
      // The RTCP wrote the conj of the data for version 1 and 2.
      if (myStManVersion < 3) {
        Array<Complex> cdata = conj(data);
        cfile->write (data.size(), cdata.data());
      } else {
        cfile->write (data.size(), data.data());
      }
      data += Complex(0.01, 0.02);
    }
    if (align2.size() > 0) {
      cfile->write (align2.size(), align2.storage());
    }
    
    if (myStManVersion < 2) {
      for (uInt j=0; j<nrbl; ++j) {
	cfile->write (nsample2.size(), nsample2.data());
	nsample2 += uShort(1);
      }
    } else {      

      for (uInt j=0; j<nrbl; ++j) {
	switch (myNrBytesPerValidSamples) {
	case 1:
	  {
	    cfile->write(nsample1.size(), nsample1.data());
	    nsample1 += uChar(1);
	  } break;
	case 2:
	  {
	    cfile->write(nsample2.size(), nsample2.data());
	    nsample2 += uShort(1);
	  } break;
	case 4:
	  {
	    cfile->write(nsample4.size(), nsample4.data());
	    nsample4 += uInt(1);
	  } break;
	}
      }
    }

    if (align3.size() > 0) {
      cfile->write (align3.size(), align3.storage());
    }
  }
  delete cfile;

  if (myStManVersion > 1) {
    TypeIO* sfile = 0;
    // create seperate file for sequence numbers if version > 1
    RegularFileIO file(RegularFile("tLofarStMan_tmp.data/table.f0seqnr"),
		       ByteIO::New);
    // Write in canonical (big endian) or local format.
    if (bigEndian) {
      sfile = new CanonicalIO(&file);
    } else {
      sfile = new RawIO(&file);
    }
    for (uInt i=0; i<nseq; ++i) {
      sfile->write (1, &i);
    }
    delete sfile;
  }
}


void checkUVW (uInt row, uInt nant, Vector<Double> uvw)
{
  // Expected outcome of UVW for antenna 0-3 and seqnr 0-1
  static double uvwVals[] = {
    0, 0, 0,
    0.423756, -127.372, 67.1947,
    0.847513, -254.744, 134.389,
    0.277918, -382.015, 201.531,
    -0.423756, 127.372, -67.1947,
    0, 0, 0,
    0.423756, -127.372, 67.1947,
    -0.145838, -254.642, 134.336,
    -0.847513, 254.744, -134.389,
    -0.423756, 127.372, -67.1947,
    0, 0, 0,
    -0.569594, -127.27, 67.1417,
    -0.277918, 382.015, -201.531,
    0.145838, 254.642, -134.336,
    0.569594, 127.27, -67.1417,
    0, 0, 0,
    0, 0, 0,
    0.738788, -127.371, 67.1942,
    1.47758, -254.742, 134.388,
    1.22276, -382.013, 201.53,
    -0.738788, 127.371, -67.1942,
    0, 0, 0,
    0.738788, -127.371, 67.1942,
    0.483976, -254.642, 134.336,
    -1.47758, 254.742, -134.388,
    -0.738788, 127.371, -67.1942,
    0, 0, 0,
    -0.254812, -127.271, 67.1421,
    -1.22276, 382.013, -201.53,
    -0.483976, 254.642, -134.336,
    0.254812, 127.271, -67.1421,
    0, 0, 0
  };
  uInt nrbl = nant*nant;
  uInt seqnr = row / nrbl;
  uInt bl = row % nrbl;
  uInt ant1 = bl % nant;
  uInt ant2 = bl / nant;
  // Only check first two time stamps and first four antennae.
  if (seqnr < 2  &&  ant1 < 4  &&  ant2 < 4) {
    AlwaysAssertExit (near(uvw[0],
                           uvwVals[3*(seqnr*16 + 4*ant1 + ant2)],
                           1e-5))
  }
}

void readTable (uInt nseq, uInt nant, uInt nchan, uInt npol,
                Double startTime, Double interval, const Complex& startValue)
{
  uInt nbasel = nant*nant;
  // Open the table and check if #rows is as expected.
  Table tab("tLofarStMan_tmp.data");
  uInt nrow = tab.nrow();
  AlwaysAssertExit (nrow = nseq*nbasel);
  AlwaysAssertExit (!tab.canAddRow());
  AlwaysAssertExit (!tab.canRemoveRow());
  AlwaysAssertExit (tab.canRemoveColumn(Vector<String>(1, "DATA")));
  // Create objects for all mandatory MS columns.
  ROArrayColumn<Complex> dataCol(tab, "DATA");
  ROArrayColumn<Float> weightCol(tab, "WEIGHT");
  ROArrayColumn<Float> wspecCol(tab, "WEIGHT_SPECTRUM");
  ROArrayColumn<Float> sigmaCol(tab, "SIGMA");
  ROArrayColumn<Double> uvwCol(tab, "UVW");
  ROArrayColumn<Bool> flagCol(tab, "FLAG");
  ROArrayColumn<Bool> flagcatCol(tab, "FLAG_CATEGORY");
  ROScalarColumn<Double> timeCol(tab, "TIME");
  ROScalarColumn<Double> centCol(tab, "TIME_CENTROID");
  ROScalarColumn<Double> intvCol(tab, "INTERVAL");
  ROScalarColumn<Double> expoCol(tab, "EXPOSURE");
  ROScalarColumn<Int> ant1Col(tab, "ANTENNA1");
  ROScalarColumn<Int> ant2Col(tab, "ANTENNA2");
  ROScalarColumn<Int> feed1Col(tab, "FEED1");
  ROScalarColumn<Int> feed2Col(tab, "FEED2");
  ROScalarColumn<Int> ddidCol(tab, "DATA_DESC_ID");
  ROScalarColumn<Int> pridCol(tab, "PROCESSOR_ID");
  ROScalarColumn<Int> fldidCol(tab, "FIELD_ID");
  ROScalarColumn<Int> arridCol(tab, "ARRAY_ID");
  ROScalarColumn<Int> obsidCol(tab, "OBSERVATION_ID");
  ROScalarColumn<Int> stidCol(tab, "STATE_ID");
  ROScalarColumn<Int> scnrCol(tab, "SCAN_NUMBER");
  ROScalarColumn<Bool> flagrowCol(tab, "FLAG_ROW");
  // Create and initialize expected data and weight.
  Array<Complex> dataExp(IPosition(2,npol,nchan));
  indgen (dataExp, startValue, Complex(0.01, 0.01));
  Array<Float> weightExp(IPosition(2,1,nchan));
  
  indgen (weightExp, Float(0),Float(1./32768));
  // Loop through all rows in the table and check the data.
  uInt row=0;
  for (uInt i=0; i<nseq; ++i) {
    
    for (uInt j=0; j<nant; ++j) {
      for (uInt k=0; k<nant; ++k) {

        // Contents must be present except for FLAG_CATEGORY.
	AlwaysAssertExit (dataCol.isDefined (row));
	AlwaysAssertExit (weightCol.isDefined (row));
        AlwaysAssertExit (wspecCol.isDefined (row));
        AlwaysAssertExit (sigmaCol.isDefined (row));
        AlwaysAssertExit (flagCol.isDefined (row));
        AlwaysAssertExit (!flagcatCol.isDefined (row));
        // Check data, weight, sigma, weight_spectrum, flag
        AlwaysAssertExit (allNear (dataCol(row), dataExp, 1e-7));
        AlwaysAssertExit (weightCol.shape(row) == IPosition(1,npol));
        AlwaysAssertExit (allEQ (weightCol(row), Float(1)));
        AlwaysAssertExit (sigmaCol.shape(row) == IPosition(1,npol));
        AlwaysAssertExit (allEQ (sigmaCol(row), Float(1)));
        Array<Float> weights = wspecCol(row);
        AlwaysAssertExit (weights.shape() == IPosition(2,npol,nchan));
        for (uInt p=0; p<npol; ++p) {

	  if (!allNear(weights(IPosition(2,p,0),IPosition(2,p,nchan-1)),weightExp, 1e-7)) {

	    std::cout << "weights: " << std::endl;
	    std::cout << weights(IPosition(2,p,0), IPosition(2,p,nchan-1)) << std::endl;
	  
	    std::cout << "weigthExp: " << std::endl;
	    std::cout << weightExp << std:: endl;

	  }
        }
        Array<Bool> flagExp (weights == Float(0));
        AlwaysAssertExit (allEQ (flagCol(row), flagExp));
        // Check ANTENNA1 and ANTENNA2

        AlwaysAssertExit (ant1Col(row) == int32(k));
        AlwaysAssertExit (ant2Col(row) == int32(j));

        dataExp += Complex(0.01, 0.02);
        weightExp += Float(1./32768);
        ++row;
      }
    }
  }
  // Check values in TIME column.
  Vector<Double> times = timeCol.getColumn();
  AlwaysAssertExit (times.size() == nrow);
  row=0;
  startTime += interval/2;
  for (uInt i=0; i<nseq; ++i) {
    for (uInt j=0; j<nbasel; ++j) {
      AlwaysAssertExit (near(times[row], startTime));
      ++row;
    }
    startTime += interval;
  }
  // Check the other columns.
  AlwaysAssertExit (allNear(centCol.getColumn(), times, 1e-13));
  AlwaysAssertExit (allNear(intvCol.getColumn(), interval, 1e-13));
  AlwaysAssertExit (allNear(expoCol.getColumn(), interval, 1e-13));
  AlwaysAssertExit (allEQ(feed1Col.getColumn(), 0));
  AlwaysAssertExit (allEQ(feed2Col.getColumn(), 0));
  AlwaysAssertExit (allEQ(ddidCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(pridCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(fldidCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(arridCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(obsidCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(stidCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(scnrCol.getColumn(), 0));
  AlwaysAssertExit (allEQ(flagrowCol.getColumn(), False));
  // Check the UVW coordinates.
  for (uInt i=0; i<nrow; ++i) {
    checkUVW (i, nant, uvwCol(i));
  }
  RefRows rownrs(0,2,1);
  Slicer slicer(IPosition(2,0,0), IPosition(2,1,1));
  cout << wspecCol(0).shape() << endl;
  Array<float> wg = wspecCol.getColumnCells (rownrs);
}

void updateTable (uInt nchan, uInt npol, const Complex& startValue)
{
  // Open the table for write.
  Table tab("tLofarStMan_tmp.data", Table::Update);
  uInt nrow = tab.nrow();
  // Create object for DATA column.
  ArrayColumn<Complex> dataCol(tab, "DATA");
  // Check we can write the column, but not change the shape.
  AlwaysAssertExit (tab.isColumnWritable ("DATA"));
  AlwaysAssertExit (!dataCol.canChangeShape());
  // Create and initialize data.
  Array<Complex> data(IPosition(2,npol,nchan));
  indgen (data, startValue, Complex(0.01, 0.01));
  // Loop through all rows in the table and write the data.
  for (uInt row=0; row<nrow; ++row) {
    dataCol.put (row, data);
    data += Complex(0.01, 0.02);
  }
}

void copyTable()
{
  Table tab("tLofarStMan_tmp.data");
  // Deep copy the table.
  tab.deepCopy ("tLofarStMan_tmp.datcp", Table::New, true);
}


int main (int argc, char* argv[])
{
  try {
    // Register LofarStMan to be able to read it back.
    LofarStMan::registerClass();
    // Get nseq, nant, nchan, npol from argv.
    uInt nseq=10;
    uInt nant=16;
    uInt nchan=256;
    uInt npol=4;
    if (argc > 1) {
      istringstream istr(argv[1]);
      istr >> nseq;
    }
    if (nseq == 0) {
      Table tab(argv[2]);
      cout << "nrow=" << tab.nrow() << endl;
      ROArrayColumn<double> uvwcol(tab, "UVW");
      cout << "uvws="<< uvwcol(0) << endl;
      cout << "uvws="<< uvwcol(1) << endl;
      cout << "uvwe="<< uvwcol(tab.nrow()-1) << endl;
    } else {
      if (argc > 2) {
        istringstream istr(argv[2]);
        istr >> nant;
      }
      if (argc > 3) {
        istringstream istr(argv[3]);
        istr >> nchan;
      }
      if (argc > 4) {
        istringstream istr(argv[4]);
        istr >> npol;
      }
      // Test the various versions.
      for (int v=1; v<4; ++v) {
        cout << "Test version " << v << endl;
        // Create the table.
        createTable (nant);
        // Write data in big-endian and check it. Align on 512.
        // Use this start time and interval to get known UVWs.
        // They are the same as used in DPPP/tUVWFlagger.
        double interval= 30.;
        double startTime = 4472025740.0 - interval*0.5;
        createData (nseq, nant, nchan, npol, startTime, interval,
                    Complex(0.1, 0.1), 512, True, v);
        readTable (nseq, nant, nchan, npol, startTime, interval,
                   Complex(0.1, 0.1));
        // Update the table and check again.
        updateTable (nchan, npol, Complex(-3.52, -20.3));
        readTable (nseq, nant, nchan, npol, startTime, interval,
                   Complex(-3.52, -20.3));
        // Write data in local format and check it. No alignment.
        createData (nseq, nant, nchan, npol, startTime, interval,
                    Complex(3.1, -5.2), 0, False, v);
        readTable (nseq, nant, nchan, npol, startTime, interval,
                   Complex(3.1, -5.2));
        // Update the table and check again.
        updateTable (nchan, npol, Complex(3.52, 20.3));
        readTable (nseq, nant, nchan, npol, startTime, interval,
                   Complex(3.52, 20.3));
        copyTable();
      }
    }
  } catch (AipsError& x) {
    cout << "Caught an exception: " << x.getMesg() << endl;
    return 1;
  } 
  return 0;                           // exit with success status
}
