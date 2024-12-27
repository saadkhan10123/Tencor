#pragma once
#include <vector>
#include <string>
#include <utility>  // For std::pair
#include "Tensor.h"

template <typename K, typename V>
class CustomHashMap {
public:
    void insert(const K& key, const V& value) {
        for (auto& pair : map) {
            if (pair.first == key) {
                pair.second = value;
                return;
            }
        }
        map.emplace_back(key, value);
    }

    bool get(const K& key, V& value) const {
        for (const auto& pair : map) {
            if (pair.first == key) {
                value = pair.second;
                return true;
            }
        }
        return false;
    }

    void clear() {
        map.clear();
    }

    const std::vector<std::pair<K, V>>& getMap() const {
        return map;
    }

private:
    std::vector<std::pair<K, V>> map;
};
