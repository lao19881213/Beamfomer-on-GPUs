//  HDF5Attributes.h: a wrapper around HDF5 attribute writing functionality
//
//  Copyright (C) 2001
//  ASTRON (Netherlands Foundation for Research in Astronomy)
//  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//
//////////////////////////////////////////////////////////////////////

#ifndef LOFAR_STORAGE_HDF5ATTRIBUTES_H
#define LOFAR_STORAGE_HDF5ATTRIBUTES_H

#ifdef HAVE_HDF5

#include <hdf5.h>

namespace LOFAR {
namespace RTCP {

// C->HDF5 translations of native types (Storage endianness)
template<typename T> hid_t h5nativeType();

template<> hid_t h5nativeType<float>()    { return H5T_NATIVE_FLOAT;  }
template<> hid_t h5nativeType<double>()   { return H5T_NATIVE_DOUBLE; }
template<> hid_t h5nativeType<unsigned>() { return H5T_NATIVE_UINT;   }
template<> hid_t h5nativeType<int>()      { return H5T_NATIVE_INT;    }
template<> hid_t h5nativeType<bool>()     { return H5T_NATIVE_CHAR;   } // assuming sizeof(bool) == 1

// C->HDF5 translations of types to use in header (ICD 003)
template<typename T> hid_t h5writeType();

template<> hid_t h5writeType<float>()    { return H5T_IEEE_F32LE; }
template<> hid_t h5writeType<double>()   { return H5T_IEEE_F64LE; }
template<> hid_t h5writeType<unsigned>() { return H5T_STD_U32LE;  }
template<> hid_t h5writeType<int>()      { return H5T_STD_I32LE;  }
template<> hid_t h5writeType<bool>()     { return H5T_STD_I32LE;  } // emulate bool with a 32-bit int

// C->HDF5 translations of types to use for data (CNProc endianness)
template<typename T> hid_t h5dataType( bool bigEndian );

template<> hid_t h5dataType<float>( bool bigEndian ) {
  return bigEndian ? H5T_IEEE_F32BE : H5T_IEEE_F32LE;
}

template<> hid_t h5dataType<LOFAR::fcomplex>( bool bigEndian ) {
  // emulate fcomplex with a 64-bit bitfield
  return bigEndian ? H5T_STD_B64BE : H5T_STD_B64LE;
}

// Autocloses hid_t types using closefunc() on destruction
class h5auto
{
public:
  h5auto( hid_t hid, hid_t (*closefunc)(hid_t) ): hid(hid), closefunc(closefunc) {}
  ~h5auto() {
    if (hid>0)
      closefunc(hid);
  }

  operator hid_t() const { return hid; }
private:
  hid_t hid;
  hid_t (*closefunc)(hid_t);
};

hid_t h5scalar()
{
  hid_t dataspace;

  dataspace = H5Screate( H5S_SCALAR );
  ASSERT( dataspace > 0 );

  return dataspace;
}

hid_t h5array( hsize_t count )
{
  hsize_t dims[1] = { count };

  hid_t dataspace = H5Screate_simple( 1, dims, NULL );
  ASSERT( dataspace > 0 );

  return dataspace;
}

hid_t h5stringType()
{
  hid_t datatype = H5Tcopy( H5T_C_S1 );
  ASSERT( datatype > 0 );

  hid_t ret = H5Tset_size( datatype, H5T_VARIABLE );
  ASSERT( ret >= 0 );

  return datatype;
}

template<typename T> void writeAttribute( hid_t loc, const char *name, T value )
{
  h5auto dataspace(h5scalar(), H5Sclose);

  h5auto attr(H5Acreate2( loc, name, h5writeType<T>(), dataspace,  H5P_DEFAULT,  H5P_DEFAULT ), H5Aclose);
  ASSERT( attr > 0 );

  hid_t ret = H5Awrite( attr, h5nativeType<T>(), &value );
  ASSERT( ret >= 0 );
}

template<typename U> void writeAttributeV( hid_t loc, const char *name, std::vector<U> value )
{
  h5auto dataspace(h5array(value.size()), H5Sclose);

  h5auto attr(H5Acreate2( loc, name, h5writeType<U>(), dataspace,  H5P_DEFAULT,  H5P_DEFAULT ), H5Aclose);
  ASSERT( attr > 0 );

  hid_t ret = H5Awrite( attr, h5nativeType<U>(), &value[0] );
  ASSERT( ret >= 0 );
}


template<> void writeAttribute( hid_t loc, const char *name, char const *value )
{
  h5auto dataspace(h5scalar(), H5Sclose);
  h5auto datatype(h5stringType(), H5Tclose);

  h5auto attr(H5Acreate2( loc, name, datatype, dataspace,  H5P_DEFAULT,  H5P_DEFAULT ), H5Aclose);
  ASSERT( attr > 0 );

  hid_t ret = H5Awrite( attr, datatype, &value );
  ASSERT( ret >= 0 );
}


template<> void writeAttribute( hid_t loc, const char *name, const std::string value )
{
  writeAttribute(loc, name, value.c_str());
}

template<> void writeAttributeV( hid_t loc, const char *name, std::vector<const char *> value )
{
  h5auto dataspace(h5array(value.size()), H5Sclose);
  h5auto datatype(h5stringType(), H5Tclose);

  h5auto attr(H5Acreate2( loc, name, datatype, dataspace,  H5P_DEFAULT,  H5P_DEFAULT ), H5Aclose);
  ASSERT( attr > 0 );

  hid_t ret = H5Awrite( attr, datatype, &value[0] );
  ASSERT( ret >= 0 );
}

template<> void writeAttributeV( hid_t loc, const char *name, std::vector<std::string> value )
{
  // convert to C-style strings
  std::vector<const char *> cstrs(value.size());
  for (unsigned i = 0; i < value.size(); i++)
    cstrs[i] = value[i].c_str();

  writeAttributeV(loc, name, cstrs);
}

}
}

#endif

#endif

