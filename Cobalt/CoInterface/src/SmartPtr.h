//# SmartPtr.h
//# Copyright (C) 2011-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: SmartPtr.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_INTERFACE_SMART_PTR_H
#define LOFAR_INTERFACE_SMART_PTR_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <cstdlib>

namespace LOFAR
{
  namespace Cobalt
  {

    template <typename T>
    class SmartPtrDelete;

    // T is the type of pointer (such as SmartPtr<int> to emulate int*)
    // D is the deletion strategy (to choose between delete/delete[]/free)
    template <typename T, class D = SmartPtrDelete<T> >
    class SmartPtr
    {
    public:
      SmartPtr(T * = NULL);
      SmartPtr(const SmartPtr<T,D> &orig); // WARNING: move semantics; orig no longer contains pointer

      ~SmartPtr();

      operator T * () const;
      T & operator * () const;
      T * operator -> () const;

      bool operator ! () const;

      SmartPtr<T,D> & operator = (T *);
      SmartPtr<T,D> & operator = (const SmartPtr<T,D> &);

      T *get();
      T *release();

    private:
      T *ptr;
    };

    // Deletion strategies
    template <typename T>
    class SmartPtrDelete
    {
    public:
      static void free( T *ptr )
      {
        delete ptr;
      }
    };

    template <typename T>
    class SmartPtrDeleteArray
    {
    public:
      static void free( T *ptr )
      {
        delete[] ptr;
      }
    };

    template <typename T>
    class SmartPtrFree
    {
    public:
      static void free( T *ptr )
      {
        ::free(ptr);
      }
    };

    template <typename T, void (*F) (T*) >
    class SmartPtrFreeFunc
    {
    public:
      static void free( T *ptr )
      {
        F(ptr);
      }
    };

    template <typename T, class D>
    inline SmartPtr<T,D>::SmartPtr(T *orig)
      :
      ptr(orig)
    {
    }

    template <typename T, class D>
    inline SmartPtr<T,D>::SmartPtr(const SmartPtr<T,D> &orig)
      :
      ptr(orig.ptr)
    {
      const_cast<T *&>(orig.ptr) = 0;
    }


    template <typename T, class D>
    inline SmartPtr<T,D>::~SmartPtr()
    {
      D::free(ptr);
    }


    template <typename T, class D>
    inline SmartPtr<T,D>::operator T * () const
    {
      return ptr;
    }


    template <typename T, class D>
    inline T &SmartPtr<T,D>::operator * () const
    {
      return *ptr;
    }


    template <typename T, class D>
    inline T *SmartPtr<T,D>::operator -> () const
    {
      return ptr;
    }


    template <typename T, class D>
    inline bool SmartPtr<T,D>::operator ! () const
    {
      return ptr == 0;
    }


    template <typename T, class D>
    inline SmartPtr<T,D> &SmartPtr<T,D>::operator = (T *orig)
    {
      D::free(ptr);
      ptr = orig;
      return *this;
    }


    template <typename T, class D>
    inline SmartPtr<T,D> &SmartPtr<T,D>::operator = (const SmartPtr<T,D> &orig)
    {
      D::free(ptr);
      ptr = orig;
      const_cast<T *&>(orig.ptr) = 0;
      return *this;
    }


    template <typename T, class D>
    inline T *SmartPtr<T,D>::get()
    {
      return ptr;
    }


    template <typename T, class D>
    inline T *SmartPtr<T,D>::release()
    {
      T *tmp = ptr;
      ptr = 0;
      return tmp;
    }


  } // namespace Cobalt
} // namespace LOFAR

#endif

