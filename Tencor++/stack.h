#pragma once
#include <iostream>

template <typename T>
class Node {
public:
    T data;          
    Node* next;

    Node(T data) {
        this->data = data;
        this->next = nullptr;
    }
};

template <typename T>
class Stack {
public:

    Stack() {
        top = nullptr;   
        size = 0;
    }

    
    ~Stack() {
        Node<T>* current = top;
        while (current != nullptr) {
            Node<T>* temp = current;
            current = current->next;
            delete temp;
        }
    }

    void push(T data) {
        Node<T>* newNode = new Node<T>(data);
        if (top == nullptr) {
            top = newNode; 
        } else {
            newNode->next = top;
            top = newNode;
        }
        size++;
    }

    T pop() {
        if (isEmpty()) {
            throw std::runtime_error("Stack underflow: Cannot pop from an empty stack");
        }
        Node<T>* temp = top;  
        T poppedData = temp->data;
        top = top->next;       
        delete temp;           
        size--;
        return poppedData;
    }

    T peek() const {
        if (isEmpty()) {
            throw std::runtime_error("Stack underflow: Cannot peek at an empty stack");
        }
        return top->data;
    }

    bool isEmpty() const {
        return top == nullptr;
    }

    int getSize() const {
        return size;
    }

    void print() const {
        Node<T>* temp = top;
        std::cout << "[";
        while (temp != nullptr) {
            std::cout << temp->data << " ";
            temp = temp->next;
        }
        std::cout << "]" << std::endl;
    }

    // Iterator class for stack traversal
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

    // Begin: Return an iterator to the top of the stack
    Iterator begin() const {
        return Iterator(top);
    }

    // End: Return an iterator to nullptr (end of the stack)
    Iterator end() const {
        return Iterator(nullptr);
    }

private:
    Node<T>* top; 
    int size;   
};
