#ifndef HASH_H
#define HASH_H

#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>

class HashTable {
public:
    // Insert a key-value pair into the hash table
    void insert(const std::string& key, const std::vector<double>& value) {
        table[key] = value;
    }

    // Remove a key-value pair from the hash table
    void remove(const std::string& key) {
        table.erase(key);
    }

    // Retrieve a value by key from the hash table
    std::vector<double> get(const std::string& key) const {
        auto it = table.find(key);
        if (it != table.end()) {
            return it->second;
        }
        throw std::runtime_error("Key not found");
    }

    // Check if a key exists in the hash table
    bool contains(const std::string& key) const {
        return table.find(key) != table.end();
    }

private:
    std::unordered_map<std::string, std::vector<double>> table;
};

#endif // HASH_H
