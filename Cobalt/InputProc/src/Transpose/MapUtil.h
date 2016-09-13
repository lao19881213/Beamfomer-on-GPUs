#ifndef LOFAR_INPUTPROC_MAPUTIL_H
#define LOFAR_INPUTPROC_MAPUTIL_H

#include <Common/LofarLogger.h>

#include <vector>
#include <map>

namespace LOFAR {

  namespace Cobalt {

    // Returns the keys of an std::map.
    template<typename K, typename V>
    std::vector<K> keys( const std::map<K, V> &m )
    {
      std::vector<K> keys;

      keys.reserve(m.size());
      for (typename std::map<K,V>::const_iterator i = m.begin(); i != m.end(); ++i) {
        keys.push_back(i->first);
      }

      return keys;
    }

    // Returns the keys of an std::map, for non-empty elements
    template<typename K, typename V>
    std::vector<K> keys_nonempty( const std::map<K, std::vector<V> > &m )
    {
      std::vector<K> keys;

      keys.reserve(m.size());
      for (typename std::map<K, std::vector<V> >::const_iterator i = m.begin(); i != m.end(); ++i) {
        if (!i->second.empty())
          keys.push_back(i->first);
      }

      return keys;
    }


    // Returns the set of unique values of an std::map.
    template<typename K, typename V>
    std::set<V> values( const std::map<K, V> &m )
    {
      std::set<V> values;

      for (typename std::map<K,V>::const_iterator i = m.begin(); i != m.end(); ++i) {
        values.insert(i->second);
      }

      return values;
    }


    // Returns the set of values of an std::map, if the values are vectors.
    template<typename K, typename V>
    std::vector<V> values( const std::map<K, std::vector<V> > &m )
    {
      std::vector<V> values;

      for (typename std::map<K, std::vector<V> >::const_iterator i = m.begin(); i != m.end(); ++i) {
        values.insert(values.end(), i->second.begin(), i->second.end());
      }

      return values;
    }


    // Returns the inverse of an std::map.
    template<typename K, typename V>
    std::map<V, std::vector<K> > inverse( const std::map<K, V> &m )
    {
      std::map<V, std::vector<K> > inverse;

      for (typename std::map<K,V>::const_iterator i = m.begin(); i != m.end(); ++i) {
        inverse[i->second].push_back(i->first);
      }

      return inverse;
    }


    // Returns the inverse of an std::map, if the values are vectors.
    template<typename K, typename V>
    std::map<V, K> inverse( const std::map<K, std::vector<V> > &m )
    {
      std::map<V, K> inverse;

      for (typename std::map<K, std::vector<V> >::const_iterator i = m.begin(); i != m.end(); ++i) {
        for (typename std::vector<V>::const_iterator j = i->second.begin(); j != i->second.end(); ++j) {
          ASSERTSTR(inverse.find(*j) == inverse.end(), "Inverting map impossible due to duplicate values");

          inverse[*j] = i->first;
        }
      }

      return inverse;
    }

  }
}

#endif

