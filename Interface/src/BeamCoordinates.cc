//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Interface/BeamCoordinates.h>
#include <Common/DataConvert.h>

#ifndef M_SQRT3
  #define M_SQRT3     1.73205080756887719000
#endif

namespace LOFAR {
namespace RTCP {

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

  for (unsigned i = 0; i < sizeof itsXYZ; i ++)
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


  for (unsigned i = 0; i < numCoordinates; i ++) {
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

  for (unsigned i = 0; i < numCoordinates; i ++)
    itsCoordinates[i].write(s);
}

BeamCoordinates& BeamCoordinates::operator+= (const BeamCoordinates &rhs)
{
  itsCoordinates.reserve(itsCoordinates.size() + rhs.size());

  for (unsigned i = 0; i < rhs.size(); i ++)
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

  for (unsigned i = 0; i < coordinates.size(); i ++)
    itsCoordinates.push_back(BeamCoord3D(coordinates[i][0], coordinates[i][1]));
}


}
}
