#pragma once 

// std
#include <functional>

namespace ud {
    // from https://stackoverflow.com/a/57595105
    // Variadic template. Rest is a pack of zero or more template type parameters
    // The ... is called the pack expansion operator, and it allows us to expand a parameter pack into 
    // a comma-separated list of parameters. Example: (hashCombine(seed, rest), ...);
    template <typename T, typename... Rest> 
    void hashCombine(std::size_t& seed, const T& v, const Rest&... rest) {
        seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        (hashCombine(seed, rest), ...);
    };
}