#ifndef FCNP_ION_H
#define FCNP_ION_H

#include <cstddef>


namespace FCNP_ION
{
  // rankInPSet is the logical rank; not the incomprehensible BG/P number!
  // ptr and size must be a multiple of 16!

  void init(bool enableInterrupts);
  void end();

  void IONtoCN_ZeroCopy(unsigned rankInPSet, unsigned channel, const void *ptr, size_t size);
  void CNtoION_ZeroCopy(unsigned rankInPSet, unsigned channel, void *ptr, size_t size);
}

#endif
