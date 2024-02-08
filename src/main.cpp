#include "first_app.hpp"

//std
#include <cstdlib>
#include <iostream>
#include <stdexcept>
/*
    UFFDEJAVU: Default Engine by JAxe in VUlkan
    or simply DEJAVU: Default Engine by JAxe in Vulkan from Uff
    ud will be its abbreviation
*/


int main() {
    ud::FirstApp app{};

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}