#include <lofar_config.h>

#if defined HAVE_BGP_ION

#include <iostream>
#include <pthread.h>
#include <sched.h>

#include <bpcore/ppc450_inlines.h>

#include <FCNP/fcnp_ion.h>


static char largeBuffer[128 * 1024 * 1024] __attribute__((aligned(16)));


static void setAffinity()
{
  cpu_set_t cpu_set;

  CPU_ZERO(&cpu_set);
  
  for (unsigned core = 1; core < 4; core ++)
    CPU_SET(core, &cpu_set);

  if (sched_setaffinity(0, sizeof cpu_set, &cpu_set) != 0) {
    std::cerr << "WARNING: sched_setaffinity failed" << std::endl;
    perror("sched_setaffinity");
  }
}


void *bidirectional(void *arg)
{
  setAffinity();

  if (arg == 0)
    FCNP_ION::IONtoCN_ZeroCopy(0, 0, largeBuffer, 128 * 1024 * 1024);
  else
    FCNP_ION::CNtoION_ZeroCopy(1, 0, largeBuffer, 128 * 1024 * 1024);

  unsigned long long start_time = _bgp_GetTimeBase();

  for (unsigned i = 0; i < 16; i ++)
    if (arg == 0)
      FCNP_ION::IONtoCN_ZeroCopy(0, 0, largeBuffer, 128 * 1024 * 1024);
    else
      FCNP_ION::CNtoION_ZeroCopy(1, 0, largeBuffer, 128 * 1024 * 1024);

  unsigned long long stop_time = _bgp_GetTimeBase();
  double time = (stop_time - start_time) / 850e6;

  std::cout << "bidirectional " << (arg ? "ION->CN" : "CN->ION") << ": " << 16ULL * 128 * 1024 * 1024 / time / 1000000 << " MB/s" << std::endl;
  return 0;
}


int main(int argc, char **argv)
{
  setAffinity();
  memset(largeBuffer, 0, sizeof largeBuffer);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " 0|1 (0 = polling, 1 = interrupts)" << std::endl;
    exit(1);
  }

  FCNP_ION::init(atoi(argv[1]) != 0);
  std::cout << "bidirectional" << std::endl;

  pthread_t thread[2];

  for (unsigned i = 0; i < 2; i ++)
    if (pthread_create(&thread[i], 0, bidirectional, (void *) i) != 0) {
      perror("pthread_create");
      exit(1);
    }

  for (unsigned i = 0; i < 2; i ++)
    if (pthread_join(thread[i], 0) != 0) {
      perror("pthread_join");
      exit(1);
    }

  std::cout << "ION -> CN (Mb/s)" << std::endl;

  unsigned long long times[28];

  for (unsigned logsize = 4; logsize <= 27; logsize ++) {
    size_t size = 1 << logsize;

    FCNP_ION::IONtoCN_ZeroCopy(0, 0, largeBuffer, size);

    unsigned long long time = 0;

    for (unsigned i = 0; i < 15; i ++) {
      //usleep(10000); // wait until the CN sent a request

      time -= _bgp_GetTimeBase();
      FCNP_ION::IONtoCN_ZeroCopy(0, 0, largeBuffer, size);
      time += _bgp_GetTimeBase();
    }

    times[logsize] = time;
  }

  for (size_t size = 16; size <= 128 * 1024 * 1024; size <<= 1) {
    for (unsigned i = 0; i < 16; i ++)
      FCNP_ION::CNtoION_ZeroCopy(0, 0, largeBuffer, size);
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

  FCNP_ION::end();

  return 0;
}

#else

int main()
{
  return 0;
}

#endif // defined HAVE_BGP_ION
