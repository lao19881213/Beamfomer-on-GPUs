#include <lofar_config.h>
#include <Storage/ExitOnClosedStdin.h>

#include <Interface/Exceptions.h>
#include <Common/SystemCallException.h>
#include <unistd.h>

namespace LOFAR {
namespace RTCP {


ExitOnClosedStdin::ExitOnClosedStdin()
:
  itsThread(this, &ExitOnClosedStdin::mainLoop, "[obs unknown] [stdinWatcherThread] ", 65535)
{
}


ExitOnClosedStdin::~ExitOnClosedStdin()
{
  itsThread.cancel();
}


void ExitOnClosedStdin::mainLoop()
{
  // an empty read on stdin means the SSH connection closed, which indicates that we should abort

  while (true) {
    fd_set fds;

    FD_ZERO(&fds);
    FD_SET(0, &fds);

    struct timeval timeval;

    timeval.tv_sec  = 1;
    timeval.tv_usec = 0;

    switch (select(1, &fds, 0, 0, &timeval)) {
      case -1 : THROW_SYSCALL("select");
      case  0 : continue;
    }

    char buf[1];
    ssize_t numbytes;
    numbytes = ::read(0, buf, sizeof buf);

    if (numbytes == 0) {
      LOG_FATAL("Lost stdin -- aborting"); // this most likely won't arrive, since stdout/stderr are probably closed as well
      exit(1);
    } else {
      // slow down reading data (IONProc will be spamming us with /dev/zero)
      if (usleep(999999) < 0)
        THROW_SYSCALL("usleep");
    }  
  }
}

}
}


