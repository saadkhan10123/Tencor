#pragma once
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>

template <typename T>
class HashTable {
public:
	HashTable() {
		// Initialize the hash table
		table = new Entry[100];
		this->size = 100;
	}

	HashTable(int size) {
		// Initialize the hash table
		table = new Entry[size];
		this->size = size;
	}

	~HashTable() {
		// Clean up the hash table
		delete[] table;
	}

	int hash(const std::string& key) const {
		// Compute the hash value for a key
		int hash = 0;
		for (char c : key) {
			hash += c;
		}
		return hash % size;
	}

	void put(const std::string& key, const T& value) {
		// Insert a key-value pair into the hash table
		int index = hash(key);

		if (table[index].key.empty()) {
			table[index].key = key;
			table[index].value = value;
			keys.push_back(key);
		}
		else {
			int originalIndex = index;
			while (!table[index].key.empty()) {
				if (table[index].key == key) {
					std::cerr << "Key already exists in the hash table" << std::endl;
					throw std::invalid_argument("Key already exists in the hash table");
				}
				index = (index + 1) % size;
				if (index == originalIndex) {
					std::cerr << "Hash table is full" << std::endl;
					throw std::runtime_error("Hash table is full");
				}
			}
			table[index].key = key;
			table[index].value = value;
			keys.push_back(key);
		}
	}

	T get(const std::string& key) const {
		// Retrieve the value associated with a key
		int index = hash(key);
		int originalIndex = index;
		while (table[index].key != key) {
			index = (index + 1) % size;
			if (index == originalIndex) {
				std::cerr << "Key not found in the hash table" << std::endl;
				throw std::invalid_argument("Key not found in the hash table");
			}
		}
		return table[index].value;
	}

	bool contains(const std::string& key) const {
		// Check if the hash table contains a key
		int index = hash(key);
		int originalIndex = index;
		while (table[index].key != key) {
			index = (index + 1) % size;
			if (index == originalIndex) {
				return false;
			}
		}
		return true;
	}

	void remove(const std::string& key) {
		// Remove a key-value pair from the hash table
		int index = hash(key);
		int originalIndex = index;
		while (table[index].key != key) {
			index = (index + 1) % size;
			if (index == originalIndex) {
				std::cerr << "Key not found in the hash table" << std::endl;
				throw std::invalid_argument("Key not found in the hash table");
			}
		}
		table[index].key.clear();
		keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
	}

	std::vector<std::string> getKeys() const {
		// Return a vector of all keys in the hash table
		return keys;
	}

	class Iterator {
	public:
		Iterator(const HashTable<T>& hashTable, size_t index = 0) : hashTable(hashTable), index(index) {}

		bool operator!=(const Iterator& other) const {
			return index != other.index;
		}

		std::pair<std::string, T> operator*() const {
			return { hashTable.keys[index], hashTable.table[hashTable.hash(hashTable.keys[index])].value };
		}

		const Iterator& operator++() {
			index++;
			return *this;
		}

	private:
		const HashTable<T>& hashTable;
		size_t index;
	};

	Iterator begin() const {
		return Iterator(*this);
	}

	Iterator end() const {
		return Iterator(*this, keys.size());
	}

private:
	struct Entry {
		std::string key;
		T value;

		Entry() : key(""), value(T()) {}
	};

	Entry* table;
	int size;
	std::vector<std::string> keys;
};
