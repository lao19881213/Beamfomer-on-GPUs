#include <lofar_config.h>

#include <StokesAsm.h>

#include <iostream>


#if defined HAVE_BGP
using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace LOFAR::TYPES;
#endif


int main()
{
#if defined HAVE_BGP
  fcomplex samples[16][2] = {{0}};

  for (int i = 0; i < 16; i ++)
    if (i != 7) {
      samples[i][0] = makefcomplex(2, 3);
      samples[i][1] = makefcomplex(4, 5);
    }

  {
    float I[16], Q[16], U[16], V[16];

    _StokesI(I, samples, 16);

    for (int i = 0; i < 16; i ++)
      std::cout << I[i] << std::endl;

    _StokesIQUV(I, Q, U, V, samples, 16);

    for (int i = 0; i < 16; i ++)
      std::cout << I[i] << ' ' << Q[i] << ' ' << U[i] << ' ' << V[i] << std::endl;
  }

  {
    float Xr = real(samples[11][0]);
    float Xi = imag(samples[11][0]);
    float Yr = real(samples[11][1]);
    float Yi = imag(samples[11][1]);

    float Xr2 = Xr * Xr;
    float Xi2 = Xi * Xi;
    float Yr2 = Yr * Yr;
    float Yi2 = Yi * Yi;

    float I = Xr2 + Xi2 + Yr2 + Yi2;
    float Q = Xr2 + Xi2 - Yr2 - Yi2;
    float U = 2 * (Xr * Yr + Xi * Yi);
    float V = 2 * (Xi * Yr - Xr * Yi);

    std::cout << I << ' ' << Q << ' ' << U << ' ' << V << std::endl;
  }
#endif

  return 0;
}
