#ifndef FCNP_CN_H
#define FCNP_CN_H

#include <cstddef>
#include <vector>


namespace FCNP_CN
{
  void init();

  // ptr and size must be a multiple of 16!
  void CNtoION_ZeroCopy(unsigned channel, const void *ptr, size_t size);
  void IONtoCN_ZeroCopy(unsigned channel, void *ptr, size_t size);
}

#endif
