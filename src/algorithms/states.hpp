#ifndef STATES_HPP
#define STATES_HPP

// Using unsigned char because it has 1 byte, so you can use each bit as a boolean, having 256 states
// since using 1 bool also uses 1 byte
// unsigned int uses 4 bytes
#include <iostream>

namespace States {
    /*
        Ex: states = 10001101
        each bit is a state
    */

    template <typename T>
    bool isIndexActive(T* states, int target){              // Check state; target is the index
        /*
            ex: 
            target = 6
            1 = 00000001; 1 << target = 01000000 
            10001101 &
            01000000
            --------
            00000000 == 01000000

            target = 3
            1 = 00000001; 1 << target = 00001000
            10001101 &
            00001000
            --------
            00001000 == 00001000
        */
        return (*states & (1 << target));
    }

    
    template<typename T>
    void activateIndex(T* states, int target) {            // Activate state
        /*
            ex: 
            target = 4
            1 = 00000001; 1 << target = 00010000 
            10001101 |
            00010000
            --------
            11011101
        */
        *states |= 1 << target;
    }

                                                            
    template<typename T>
    void uniquelyActivateIndex(T* states, int target) {     // No other state is active
        /*
            ex: 
            target = 4
            1 = 00000001; 1 << target = 00010000 
            10001101 & (active state)
            00010000
            --------
            00010000
        */
        activateIndex<T>(states, target);                   // Activate state
        *states &= (1 << target);                           // Deactivate all other states
    }

    // deactivate state
    template<typename T>
    void deactivateIndex(T* states, int target) {
        /*
            ex: 
            target = 4
            1 = 00000001; 1 << target = 00010000 ; ~(1 << target) = 11101111
            10001101 &
            11101111
            --------
            10001101
        */
        *states &= ~(1 << target);
    }

    // toggle state
    template<typename T>
    void toggleIndex(T* states, int target) {
        /*
            ex: 
            target = 4
            1 = 00000001; 1 << target = 00010000; ^ = XOR; ^(1 << target)
            10001101 ^
            00010000
            --------
            10011101
        */
        *states ^= (1 << target);
    }
    // check state
    template<typename T>
    bool isActive(T* states, T state) {
        return (*states & state) == state;
    }
 
    // activate state
    template<typename T>
    void activate(T* states, T state) {
        *states |= state;
    }
 
    // uniquely activate state (no others can be active)
    template<typename T>
    void uniquelyActivate(T* states, T state) {
        *states &= state;
    }
 
    // deactivate state
    template<typename T>
    void deactivate(T* states, T state) {
        *states &= ~state;
    }
 
    // toggle state
    template<typename T>
    void toggle(T* states, T state) {
        *states ^= state;
    }

};

#endif