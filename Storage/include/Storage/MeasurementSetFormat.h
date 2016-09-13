//#  Format.h: defines the format of the RAW datafile
//#
//#  Copyright (C) 2009
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  $Id: Storage_main.cc 12953 2009-03-26 17:10:42Z nieuwpoort $

#ifndef LOFAR_STORAGEFORMAT_H
#define LOFAR_STORAGEFORMAT_H

#include <Common/Thread/Mutex.h>
#include <Interface/Parset.h>
#include <Interface/SmartPtr.h>

#include <casa/aips.h>
#include <casa/Utilities/DataType.h>
#include <casa/Arrays/IPosition.h>

#include <Storage/Format.h>

#include <string>

//# Forward Declarations
namespace casa
{
  class TableDesc;
  class MPosition;
  template<class T> class Block;
}


namespace LOFAR {
  //# Forward Declarations
  class MSLofar;

namespace RTCP {

class MeasurementSetFormat : public Format
{
  public:
	    MeasurementSetFormat(const Parset &, uint32 alignment = 1);
    virtual ~MeasurementSetFormat();

    virtual void addSubband(const string MSname, unsigned subband, bool isBigEndian);

    // casacore/measurementset mutex
    static Mutex sharedMutex;

  private:
    const Parset &itsPS;

    const vector<string> stationNames;
    const vector<double> antPos;

    const unsigned itsNrAnt;
    uint32 itsNrTimes;

    double itsStartTime;
    double itsTimeStep;


    SmartPtr<MSLofar> itsMS;

    const uint32 itsAlignment;

    void createMSTables(const string &MSname, unsigned subband);
    void createMSMetaFile(const string &MSname, unsigned subband, bool isBigEndian);

    void fillFeed();
    void fillAntenna(const casa::Block<casa::MPosition>& antMPos);
    void fillField(unsigned subarray);
    void fillPola();
    void fillDataDesc();
    void fillSpecWindow(unsigned subband);
    void fillObs(unsigned subarray);
    void fillHistory();
};

} //RTCP
} //LOFAR
#endif // LOFAR_STORAGEFORMAT_H
