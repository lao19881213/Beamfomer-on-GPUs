#ifndef LOFAR_INTERFACE_BEAMCOORDINATES_H
#define LOFAR_INTERFACE_BEAMCOORDINATES_H

#include <Interface/MultiDimArray.h>
#include <Common/DataConvert.h>
#include <Stream/Stream.h>
#include <vector>
#include <cmath>
#include <ostream>

namespace LOFAR {
namespace RTCP {

// Beam coordinates are offsets for pencil beams (tied array beams) relative to the center
// of the station/beamformer beam.

class BeamCoord3D {
  public:
    BeamCoord3D(double ra, double dec) {
      itsXYZ[0] = ra;
      itsXYZ[1] = dec;
      itsXYZ[2] = sqrt(1.0 - ra * ra - dec * dec);
    /*
      // (ra,dec) is a spherical direction, but the station positions
      // and phase centers are cartesian (x,y,z with origin close to the geocenter).
      // Spherical coordinates are converted to cartesian as follows:
      //
      // 	phi = .5pi - DEC, theta = RA (in parset: angle1=RA, angle2=DEC)
      //        rho = 1 (distance), since we need to construct a unit vector
      //
      //        then: x = rho*sin(phi)*cos(theta), y = rho*sin(phi)*sin(theta), z = rho*cos(theta)
      //
      // NOTE: The use of the letters phi and theta differ or are swapped between sources.

      // in this case, phi is relative to the original beam, so .5pi is already compensated for. The
      // direction of DEC is still important, so we have to use phi = -dec to get the proper relative change
      // in angle.
      const double phi = -dec;
      const double theta = ra;

      itsXYZ[0] = sin(phi)*cos(theta);
      itsXYZ[1] = sin(phi)*sin(theta);
      itsXYZ[2] = cos(theta);
    */  
    }

    BeamCoord3D(double x, double y, double z) {
      itsXYZ[0] = x;
      itsXYZ[1] = y;
      itsXYZ[2] = z;
    }

    BeamCoord3D(const double xyz[3]) {
      itsXYZ[0] = xyz[0];
      itsXYZ[1] = xyz[1];
      itsXYZ[2] = xyz[2];
    }

    BeamCoord3D(std::vector<double> xyz) {
      itsXYZ[0] = xyz[0];
      itsXYZ[1] = xyz[1];
      itsXYZ[2] = xyz[2];
    }

    inline BeamCoord3D& operator-= (const BeamCoord3D &rhs)
    {
      itsXYZ[0] -= rhs.itsXYZ[0];
      itsXYZ[1] -= rhs.itsXYZ[1];
      itsXYZ[2] -= rhs.itsXYZ[2];

      return *this;
    }

    inline BeamCoord3D& operator+= (const BeamCoord3D &rhs)
    {
      itsXYZ[0] += rhs.itsXYZ[0];
      itsXYZ[1] += rhs.itsXYZ[1];
      itsXYZ[2] += rhs.itsXYZ[2];

      return *this;
    }

    inline BeamCoord3D& operator*= (double a)
    {
      itsXYZ[0] *= a;
      itsXYZ[1] *= a;
      itsXYZ[2] *= a;

      return *this;
    }

    inline double operator[] (unsigned i) const
    {
      return itsXYZ[i];
    }

    inline double &operator[] (unsigned i)
    {
      return itsXYZ[i];
    }

    friend double operator* (const BeamCoord3D &lhs, const BeamCoord3D &rhs);
    friend std::ostream& operator<< (std::ostream &os, const BeamCoord3D &c);

    void read(Stream *);
    void write(Stream *) const;

  private:
    double itsXYZ[3];
};


// BeamCoordinates are coordinates of the pencil beams that need to
// be formed. Each coordinate is a normalised vector, relative to the
// center beam.
//
// The center beam has to be included as the first coordinate of (0,0,1).
class BeamCoordinates
{
  public:
    BeamCoordinates() {}
    BeamCoordinates(const std::vector<BeamCoord3D> &coordinates): itsCoordinates(coordinates) {}
    BeamCoordinates(const Matrix<double> &coordinates);

    inline std::vector<BeamCoord3D>& getCoordinates() 
    { return itsCoordinates; }

    inline size_t size() const
    { return itsCoordinates.size(); }

    inline const BeamCoord3D &operator[] (unsigned nr) const
    { return itsCoordinates[nr]; }

    void read(Stream *s);
    void write(Stream *s) const;

    BeamCoordinates& operator += (const BeamCoordinates &rhs);
    BeamCoordinates& operator += (const BeamCoord3D &rhs);

    friend std::ostream& operator<< (std::ostream &os, const BeamCoordinates &c);

private:
    std::vector<BeamCoord3D>  itsCoordinates;
};

inline double operator* (const BeamCoord3D &lhs, const BeamCoord3D &rhs)
{
  double sum = 0;

  sum += lhs.itsXYZ[0] * rhs.itsXYZ[0];
  sum += lhs.itsXYZ[1] * rhs.itsXYZ[1];
  sum += lhs.itsXYZ[2] * rhs.itsXYZ[2];
  return sum;
}

inline BeamCoord3D& operator- (const BeamCoord3D &lhs, const BeamCoord3D &rhs)
{
  return BeamCoord3D(lhs) -= rhs;
}

inline BeamCoord3D& operator+ (const BeamCoord3D &lhs, const BeamCoord3D &rhs)
{
  return BeamCoord3D(lhs) += rhs;
}

inline BeamCoord3D& operator* (double a, const BeamCoord3D &rhs)
{
  return BeamCoord3D(rhs) *= a;
}

inline BeamCoord3D& operator* (const BeamCoord3D &lhs, double a)
{
  return BeamCoord3D(lhs) *= a;
}

inline std::ostream& operator << (std::ostream& os, const BeamCoord3D &c)
{
  return os << "(" << c.itsXYZ[0] << "," << c.itsXYZ[1] << "," << c.itsXYZ[2] << ")";
}

inline std::ostream& operator << (std::ostream &os, const BeamCoordinates &c)
{
  return os << c.itsCoordinates;
}

}
}

#endif
