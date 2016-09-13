#include <lofar_config.h>

#include <FIR_Asm.h>
#include <PPF.h>
#include <Common/Timer.h>

#include <iostream>


#if !defined PPF_C_IMPLEMENTATION && !defined FIR_C_IMPLEMENTATION
using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace LOFAR::TYPES;
#endif

#define SIZE	131072

int main()
{
#if !defined PPF_C_IMPLEMENTATION && !defined FIR_C_IMPLEMENTATION
  {
    i16complex in[SIZE] = {
      makei16complex(0x0100, 0x0200),
      makei16complex(0x0300, 0x0400),
      makei16complex(0x0500, 0x0600),
    };

    in[SIZE - 1] = makei16complex(0x0700, 0x0801);

    fcomplex   out[SIZE];

    NSTimer timer("little endian i16complex -> float", true);
    timer.start();
    _convert(out, in, SIZE);
    timer.stop();

    std::cout << out[0] << ' ' << out[1] << ' ' << out[2] << ' ' << out[SIZE - 1] << std::endl;
  }

  {
    PPF<i8complex>::initConstantTable();

    i8complex in[SIZE] = {
      makei8complex(1, 2),
      makei8complex(3, 4),
      makei8complex(5, 6),
    };

    in[SIZE - 1] = makei8complex(7, 8);

    fcomplex   out[SIZE];

    NSTimer timer("little endian i8complex -> float", true);
    timer.start();
    _convert(out, in, SIZE);
    timer.stop();

    std::cout << out[0] << ' ' << out[1] << ' ' << out[2] << ' ' << out[SIZE - 1] << std::endl;
  }

  {
    PPF<i4complex>::initConstantTable();

    i4complex in[SIZE] = {
      makei4complex(0.5, 1.5),
      makei4complex(2.5, 3.5),
      makei4complex(4.5, 5.5),
    };

    in[SIZE - 1] = makei4complex(-1.5, -0.5);

    fcomplex   out[SIZE];

    NSTimer timer("little endian i4complex -> float", true);
    timer.start();
    _convert(out, in, SIZE);
    timer.stop();

    std::cout << out[0] << ' ' << out[1] << ' ' << out[2] << ' ' << out[SIZE - 1] << std::endl;
  }
#endif
  return 0;
}
