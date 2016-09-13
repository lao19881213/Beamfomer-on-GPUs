//#  Format.h: Virtual baseclass 
//#
//#  Copyright (C) 2009
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  $Id: $

#ifndef LOFAR_STORAGE_FORMAT_H
#define LOFAR_STORAGE_FORMAT_H

#include <string>

namespace LOFAR {
namespace RTCP {

class Format
{
  public:
    virtual ~Format();
    
    virtual void addSubband(const std::string MSname, unsigned subband, bool isBigEndian) = 0;
};

} // namespace RTCP
} // namespace LOFAR

#endif
