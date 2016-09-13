#include <lofar_config.h>

#if defined HAVE_BGP_CN

#include <mpi.h>
#include <cstring>
#include <iostream>

#include <bpcore/ppc450_inlines.h>

#include <FCNP/fcnp_cn.h>


static char largeBuffer[128 * 1024 * 1024] __attribute__((aligned(16)));


int main(int argc, char **argv)
{
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  FCNP_CN::init();

  if (rank <= 1) {
    memset(largeBuffer, 0, sizeof largeBuffer);

    if (rank == 0)
      std::cout << "bidirectional" << std::endl;

    for (unsigned i = 0; i < 17; i ++)
      if (rank == 0)
	FCNP_CN::IONtoCN_ZeroCopy(0, largeBuffer, 128 * 1024 * 1024);
      else
	FCNP_CN::CNtoION_ZeroCopy(0, largeBuffer, 128 * 1024 * 1024);
  }

  if (rank == 0) {
    unsigned long long times[28];

    std::cout << "CN -> ION (Mb/s)" << std::endl;

    for (size_t size = 16; size <= 128 * 1024 * 1024; size <<= 1)
       for (unsigned i = 0; i < 16; i ++)
	FCNP_CN::IONtoCN_ZeroCopy(0, largeBuffer, size);

    for (unsigned logsize = 4; logsize <= 27; logsize ++) {
      size_t size = 1 << logsize;
      FCNP_CN::CNtoION_ZeroCopy(0, largeBuffer, size);

      unsigned long long time = 0;

      for (unsigned i = 1; i < 16; i ++) {
	//usleep(10000); // wait until ION is ready to receive

	time -= _bgp_GetTimeBase();
	FCNP_CN::CNtoION_ZeroCopy(0, largeBuffer, size);
	time += _bgp_GetTimeBase();
      }

      times[logsize] = time;
    }

    for (unsigned logsize = 4; logsize <= 27; logsize ++) {
      size_t size = 1 << logsize;
      std::cout << size << ' ' << (15 * 8.0 * size / (times[logsize] / 850.0)) << std::endl;
    }

    double smallMsg = times[4] / 850e6 / 15;
    double largeMsg = times[27] / 850e6 / 15;

    double bandwidth = (128 * 1024 * 1024 - 16) / (largeMsg - smallMsg);
    double latency = smallMsg - 16 / bandwidth;

    std::cout << "latency = " << latency * 1e6 << " us, bandwidth = " << bandwidth / 1e9 << " GB/s" << std::endl;
    abort(); // force exit
  }

  MPI_Finalize();

  return 0;
}

#else

int main()
{
  return 0;
}

#endif // defined HAVE_BGP_CN
