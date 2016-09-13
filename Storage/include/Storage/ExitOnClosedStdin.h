//#  ExitOnClosedStdin.h: Exit program if stdin closes.
//#
//#  Copyright (C) 2009
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  $Id: $

#ifndef LOFAR_STORAGE_EXIT_ON_CLOSED_STDIN_H
#define LOFAR_STORAGE_EXIT_ON_CLOSED_STDIN_H

#include <Common/Thread/Thread.h>

namespace LOFAR {
namespace RTCP {

class ExitOnClosedStdin
{
  public:
    ExitOnClosedStdin();
    ~ExitOnClosedStdin();

  private:
    void   mainLoop();
    Thread itsThread;
};

} // namespace RTCP
} // namespace LOFAR

#endif

