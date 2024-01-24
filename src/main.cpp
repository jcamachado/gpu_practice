#include "first_app.hpp"

//std
#include <cstdlib>
#include <iostream>
#include <stdexcept>
/*
    UFFDEJAVU Default Engine by JAxe in VUlkan
    ud will be its abbreviation
*/


int main() {
    uffdejavu::FirstApp app{};

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}