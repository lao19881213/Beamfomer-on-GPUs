//# OpenMP_Lock.h
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: OpenMP_Lock.h 24984 2013-05-21 16:18:43Z amesfoort $

#ifndef LOFAR_GPUPROC_OPENMP_LOCK_H
#define LOFAR_GPUPROC_OPENMP_LOCK_H

#include <omp.h>

class OMP_Lock
{
public:
  OMP_Lock()
  {
    omp_init_lock(&omp_lock);
  }

  ~OMP_Lock()
  {
    omp_destroy_lock(&omp_lock);
  }

  void lock()
  {
    omp_set_lock(&omp_lock);
  }

  void unlock()
  {
    omp_unset_lock(&omp_lock);
  }

private:
  omp_lock_t omp_lock;
};


class OMP_ScopedLock
{
public:
  OMP_ScopedLock(OMP_Lock &omp_lock)
    :
    omp_lock(omp_lock)
  {
    omp_lock.lock();
  }

  ~OMP_ScopedLock()
  {
    omp_lock.unlock();
  }

private:
  OMP_Lock &omp_lock;
};

#endif

