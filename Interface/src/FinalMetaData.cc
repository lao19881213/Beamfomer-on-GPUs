//# FinalMetaData.cc:
//#
//#  Copyright (C) 2001
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: $

#include <lofar_config.h>

#include <Interface/FinalMetaData.h>
#include <Common/LofarTypes.h>
#include <Common/DataConvert.h>

namespace LOFAR {
namespace RTCP {

// TODO: Export these functions to be globally available

template<typename T> class StreamWriter {
  public:
    static void write( Stream &s, const T &data );
    static void read( Stream &s, T &data );
};

template<typename T> class StreamWriter< std::vector<T> > {
  public:
    static void write( Stream &s, const std::vector<T> &data );
    static void read( Stream &s, std::vector<T> &data );
};

template<> void StreamWriter<size_t>::write( Stream &s, const size_t &data )
{
  uint64 raw = data;

#if !defined WORDS_BIGENDIAN
  dataConvert(LittleEndian, &raw, 1);
#endif

  s.write(&raw, sizeof raw);
}

template<> void StreamWriter<size_t>::read( Stream &s, size_t &data )
{
  uint64 raw_nr;

  s.read(&raw_nr, sizeof raw_nr);

#if !defined WORDS_BIGENDIAN
  dataConvert(LittleEndian, &raw_nr, 1);
#endif

  data = raw_nr;
}

template<> void StreamWriter<std::string>::write( Stream &s, const std::string &data )
{
  size_t len = data.size();

  StreamWriter<size_t>::write(s, len);

  if (len > 0)
    s.write(data.data(), len);
}

template<> void StreamWriter<std::string>::read( Stream &s, std::string &data )
{
  size_t len;

  StreamWriter<size_t>::read(s, len);

  std::vector<char> buffer(len);
  s.read(&buffer[0], len);

  data.assign(&buffer[0], len);
}

template<typename T> void StreamWriter< std::vector<T> >::write( Stream &s, const std::vector<T> &data )
{
  size_t len = data.size();

  StreamWriter<size_t>::write(s, len);

  for (size_t i = 0; i < len; ++i)
    StreamWriter<T>::write(s, data[i]);
}

template<typename T> void StreamWriter< std::vector<T> >::read( Stream &s, std::vector<T> &data )
{
  size_t len;

  StreamWriter<size_t>::read(s, len);

  data.resize(len);

  for (size_t i = 0; i < len; ++i)
    StreamWriter<T>::read(s, data[i]);
}

template<> void StreamWriter<struct FinalMetaData::BrokenRCU>::write( Stream &s, const struct FinalMetaData::BrokenRCU &data )
{
  StreamWriter<std::string>::write(s, data.station);
  StreamWriter<std::string>::write(s, data.type);
  StreamWriter<size_t>     ::write(s, data.seqnr);
  StreamWriter<std::string>::write(s, data.time);
}

template<> void StreamWriter<struct FinalMetaData::BrokenRCU>::read( Stream &s, struct FinalMetaData::BrokenRCU &data )
{
  StreamWriter<std::string>::read(s, data.station);
  StreamWriter<std::string>::read(s, data.type);
  StreamWriter<size_t>     ::read(s, data.seqnr);
  StreamWriter<std::string>::read(s, data.time);
}

void FinalMetaData::write(Stream &s)
{
  StreamWriter< std::vector<struct BrokenRCU> >::write(s, brokenRCUsAtBegin);
  StreamWriter< std::vector<struct BrokenRCU> >::write(s, brokenRCUsDuring);
}

void FinalMetaData::read(Stream &s)
{
  StreamWriter< std::vector<struct BrokenRCU> >::read(s, brokenRCUsAtBegin);
  StreamWriter< std::vector<struct BrokenRCU> >::read(s, brokenRCUsDuring);
}

std::ostream& operator<<(std::ostream& os, const struct FinalMetaData::BrokenRCU &rcu)
{
  os << "(" << rcu.station << " " << rcu.type << " " << rcu.seqnr << " " << rcu.time << ")";

  return os;
}

std::ostream& operator<<(std::ostream& os, const FinalMetaData &finalMetaData)
{
  os << "Broken RCUs at begin of obs: ";

  for (size_t i = 0; i < finalMetaData.brokenRCUsAtBegin.size(); i++) {
    const struct FinalMetaData::BrokenRCU &rcu = finalMetaData.brokenRCUsAtBegin[i];

    if (i > 0)
      os << ", ";

    os << rcu;
  }

  os << " Broken RCUs during obs: ";

  for (size_t i = 0; i < finalMetaData.brokenRCUsDuring.size(); i++) {
    const struct FinalMetaData::BrokenRCU &rcu = finalMetaData.brokenRCUsDuring[i];

    if (i > 0)
      os << ", ";

    os << rcu;
  }

  return os;
}

}
}
