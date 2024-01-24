#ifndef LIST_HPP
#define LIST_HPP

#include <algorithm>

namespace List {
    // Find index of item in vector (list)
    template <typename T>
    int getIndexOf(std::vector<T> v, T x){
        // Iterator(end of vector) - Iterator(beginning of vector) = index of item
        // Kinda N - 0 = N
        return std::find(v.begin(), v.end(), x) - v.begin(); 
    }

    // Test if list contains item
    template <typename T>
    bool contains(std::vector<T> v, T x){
        // Iterator(a) != Iterator(N) = has item in position a, and a < N
        return std::find(v.begin(), v.end(), x) != v.end();
    }
};

#endif