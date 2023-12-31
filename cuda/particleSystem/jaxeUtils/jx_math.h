#include <random>

using namespace std;

// Math utils
int randomInt(int min, int max){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distr(min, max);
    return distr(gen);
}
