//# Stream.cc
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
//# $Id: Stream.cc 26956 2013-10-14 09:49:52Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

//# Includes
#include <CoInterface/Stream.h>

#include <ctime>
#include <vector>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <Common/StringUtil.h>
#include <Common/LofarLogger.h>
#include <CoInterface/Exceptions.h>
#include <Stream/FileStream.h>
#include <Stream/NullStream.h>
#include <Stream/SocketStream.h>
#include <Stream/PortBroker.h>
#include <Stream/NamedPipeStream.h>


using boost::format;
using namespace std;

namespace LOFAR
{
  namespace Cobalt
  {

    Stream *createStream(const string &descriptor, bool asServer, time_t deadline)
    {
      vector<string> split = StringUtil::split(descriptor, ':');

      if (deadline > 0 && deadline <= std::time(0))
        THROW(SocketStream::TimeOutException, "Deadline already passed at start");

      if (descriptor == "null:")
        return new NullStream;
      else if (split.size() == 3 && split[0] == "udp")
        return new SocketStream(split[1].c_str(), boost::lexical_cast<unsigned short>(split[2]), SocketStream::UDP, asServer ? SocketStream::Server : SocketStream::Client, deadline);
      else if (split.size() == 3 && split[0] == "tcp")
        return new SocketStream(split[1].c_str(), boost::lexical_cast<unsigned short>(split[2]), SocketStream::TCP, asServer ? SocketStream::Server : SocketStream::Client, deadline);
      else if (split.size() == 3 && split[0] == "udpkey")
        return new SocketStream(split[1].c_str(), 0, SocketStream::UDP, asServer ? SocketStream::Server : SocketStream::Client, deadline, split[2].c_str());
      else if (split.size() == 4 && split[0] == "tcpbroker")
        return asServer ? static_cast<Stream*>(new PortBroker::ServerStream(split[3])) : static_cast<Stream*>(new PortBroker::ClientStream(split[1], boost::lexical_cast<unsigned short>(split[2]), split[3]));
      else if (split.size() == 3 && split[0] == "tcpkey")
        return new SocketStream(split[1].c_str(), 0, SocketStream::TCP, asServer ? SocketStream::Server : SocketStream::Client, deadline, split[2].c_str());
      else if (split.size() > 1 && split[0] == "file") {
        // don't use split[1] to allow : in filenames
        const string filename = descriptor.substr(5);
        return asServer ? new FileStream(filename.c_str()) : new FileStream(filename.c_str(), 0666);
      } else if (split.size() == 2 && split[0] == "pipe")
        return new NamedPipeStream(split[1].c_str(), asServer);
      else if (split.size() == 2)
        return new SocketStream(split[0].c_str(), boost::lexical_cast<unsigned short>(split[1]), SocketStream::UDP, asServer ? SocketStream::Server : SocketStream::Client, deadline);
      else if (split.size() == 1)
        return asServer ? new FileStream(split[0].c_str()) : new FileStream(split[0].c_str(), 0666);
      else
        THROW(CoInterfaceException, string("unrecognized connector format: \"" + descriptor + '"'));
    }


    uint16 storageBrokerPort(int observationID)
    {
      return 7000 + observationID % 1000;
    }


    string getStorageControlDescription(int observationID, int rank)
    {
      return str(format("[obs %d rank %d] control") % observationID % rank);
    }


    string getStreamDescriptorBetweenIONandStorage(const Parset &parset, OutputType outputType, unsigned streamNr)
    {
      string host = parset.getHostName(outputType, streamNr);
      uint16 port = storageBrokerPort(parset.observationID());
      return str(format("tcpbroker:%s:%u:ion-storage-obs-%u-type-%u-stream-%u") % host % port % parset.observationID() % outputType % streamNr);
    }

  } // namespace Cobalt
} // namespace LOFAR

