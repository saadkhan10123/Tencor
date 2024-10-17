#pragma once
#include <iostream>

template <typename T>
class Node {
public:
    T data;
    Node* next;
    Node* prev;

    Node(T data) {
        this->data = data;
        this->next = nullptr;
        this->prev = nullptr;
    }
};

template <typename T>
class List {
public:
    List() {
        head = nullptr;
        tail = nullptr;
        size = 0;
    }

    List(T data) {
        head = new Node<T>(data);
        tail = head;
        size = 1;
    }

    ~List() {
        Node<T>* current = head;
        Node<T>* next = nullptr;

        while (current != nullptr) {
            next = current->next;
            delete current;
            current = next;
        }
    }

    void append(T data) {
        Node<T>* newNode = new Node<T>(data);

        if (head == nullptr) {
            head = newNode;
            tail = newNode;
            size = 1;
            return;
        }

        tail->next = newNode;
        newNode->prev = tail;
        tail = newNode;
        size++;
    }

    void print() const {
        Node<T>* temp = head;
        std::cout << "[";
        while (temp != nullptr) {
            std::cout << temp->data << " ";
            temp = temp->next;
        }
        std::cout << "]" << std::endl;
    }

    T get(int index) const {
        Node<T>* temp = head;
        for (int i = 0; i < index; i++) {
            temp = temp->next;
        }
        return temp->data;
    }

    int getSize() const {
        return this->size;
    }

    // Iterator class
    class Iterator {
    public:
        Iterator(Node<T>* node) : current(node) {}

        T& operator*() const {
            return current->data;
        }

        Iterator& operator++() {
            current = current->next;
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return current != other.current;
        }

    private:
        Node<T>* current;
    };

    Iterator begin() const {
        return Iterator(head);
    }

    Iterator end() const {
        return Iterator(nullptr);
    }

private:
    Node<T>* head;
    Node<T>* tail;
    int size;
};
