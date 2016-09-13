//# Ranges.cc
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
//# $Id: Ranges.cc 25606 2013-07-09 19:36:21Z mol $

#include <lofar_config.h>

#include "Ranges.h"

#include <algorithm>
#include <sstream>
#include <Common/LofarLogger.h>

namespace LOFAR
{
  namespace Cobalt
  {

    std::ostream& operator<<( std::ostream &str, const Ranges &r )
    {
      for (struct Ranges::const_iterator i = r.begin(); i != r.end(); ++i) {
       if (i != r.begin())
         str << " ";

       str << "[" << i.from() << ", " << i.to() << ")";
      }

      return str;
    }

    void Ranges::dump() const {
      stringstream s;
      for (struct Range *i = _begin; i != _end; ++i) {
        if (head == i) {
          s << "HEAD -> ";
        }
        s << "[" << i->from << ", " << i->to << ") ";
      }

      LOG_DEBUG_STR(s.str());
    }

    Ranges::Ranges()
      :
      create(false),
      len(0),
      ranges(0),
      _begin(0),
      _end(_begin),
      head(_begin),
      minHistory(0)
    {
    }

    Ranges::Ranges( void *data, size_t numBytes, value_type minHistory, bool create )
      :
      create(create),
      len(numBytes / sizeof *ranges),
      ranges(create ? new(data)Range[len] : static_cast<Range*>(data)),
      _begin(&ranges[0]),
      _end(&ranges[len]),
      head(_begin),
      minHistory(minHistory)
    {
      ASSERT( len > 0 );
    }

    Ranges::~Ranges()
    {
      if (create)
        for (struct Range *i = _begin; i != _end; ++i)
          i->~Range();
    }

    void Ranges::clear()
    {
      for (struct Range *i = _begin; i != _end; ++i) {
        // erase; delete 'to' first!
        i->to = 0;
        i->from = 0;
      }

      head = _begin;
    }

    void Ranges::excludeBefore( value_type to )
    {
      for (struct Range *i = _begin; i != _end; ++i) {
        if (i->to <= to) {
          // erase; delete 'to' first!
          i->to = 0;
          i->from = 0;
          continue;
        }

        if (i->from < to) {
          // shorten
          i->from = to;
        }
      }
    }

    void Ranges::remove( Range *i )
    {
      if (i != head) {
        // shift entries until `head'
        do {
          Range *prev = i;
          i = next_rr(i);

          // copy 'from' first to invalidate 'prev' entry
          prev->from = i->from;

          // copy 'to' to complete (and validate) 'prev' entry
          prev->to = i->to;
        } while (i != head);
      }

      // i == head: clear head and move head back one spot
      i->to = 0;
      i->from = 0;
      head = prev_rr(i);
    }

    bool Ranges::insert_empty( Range *i, value_type to )
    {
      if (i->to == 0) {
        // spot already free
        return true;
      }

      Range *tail = next_rr(head);

      if (tail->to != 0 && (to < minHistory || tail->to >= to - minHistory)) {
        // no room
        return false;
      }

      // shift entries from 'head' down to 'i'
      for(Range *b = head, *next = tail; next != i; next = b, b = prev_rr(b)) {
        // copy 'to' first to invalidate 'b' entry
        next->to = b->to;

        // copy 'from' to complete (and validate) 'b' entry
        next->from = b->from;
      }

      // shift head as well
      head = tail;

      // clean our spot
      i->to = 0;
      i->from = 0;
      return true;
    }

    bool Ranges::include( value_type from, value_type to )
    {
      ASSERTSTR( from < to, from << " < " << to );

      if (head->to == 0) {
        /*
         * Ranges are empty, fill *head
         */

        // *head is unused, set 'from' first!
        head->from = from;
        head->to = to;
        return true;
      }

      if (head->to == from) {
        /*
         * In-order arrival, next packet arrived
         */

        // *head can be extended
        head->to = to;
        return true;
      }

      if (head->to < from) {
        /*
         * In-order arrival, but packet loss
         */

        // new range is needed
        struct Range * const next = next_rr(head);

        if (next->to == 0 || (to > minHistory && next->to < to - minHistory)) {
          // range at 'next' is either unused or old enough to toss away
          next->from = from;
          next->to = to;

          head = next;
          return true;
        }

        // no room -- discard
        return false;
      }

      ASSERT(head->to > from);

      /*
       * Out-of-order arrival
       */

      // scan all ranges to see where we fit in
      for (struct Range *i = _begin; i != _end; ++i) {
        if (i->to == 0)
          continue;

        // we shouldn't fall into an already existing range,
        // or we'd be a duplicate packet
        ASSERT(to <= i->from || from >= i->to);

        if (i->to == from) {
          // *i can be extended
          i->to = to;

          struct Range *next = next_rr(i);

          if (i->to == next->from) {
            // merge *i and *next
            i->to = next->to;
            remove(next);
          }

          return true;
        } else if (i->from == to) {
          // *i can be extended
          i->from = from;

          struct Range *prev = prev_rr(i);

          if (prev->to == i->from) {
            // merge *prev and *i
            prev->to = i->to;
            remove(i);
          }

          return true;
        }
      }

      // Out-of-order, but we need a new range!
      for (struct Range *i = _begin; i != _end; ++i) {
        struct Range *prev = prev_rr(i);

        if (prev->to < from && to < i->from) {
          // we fit in right here.

          if (!insert_empty(i, to))
            // couldn't free a spot
            return false;

          // insert at the new spot
          i->to = to;
          i->from = from;
          return true;
        }
      }

      // No room
      return false;
    }

    bool Ranges::anythingBetween( value_type first, value_type last ) const
    {
      for(const_iterator i = begin(); i != end(); ++i) {
        value_type from = std::max( i.from(), first );
        value_type to = std::min( i.to(), last );

        if (from < to)
          return true;
      }

      return false;
    }

    BufferSettings::flags_type Ranges::sparseSet( value_type first, value_type last ) const
    {
      BufferSettings::flags_type result;

      if (first >= last)
        return result;

      for(const_iterator i = begin(); i != end(); ++i) {
        value_type from = std::max( i.from(), first );
        value_type to = std::min( i.to(), last );

        if (from < to)
          result.include(from, to);
      }

      return result;
    }

  }
}

