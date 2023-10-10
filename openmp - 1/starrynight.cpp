#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int nthreads, tid;
  printf("Welcome to the Starry Night printer!");
  char grid[10000];
  char star = '*';
  char space = ' ';
  char newline = '\n';
#pragma omp parallel
  {
    tid = omp_get_thread_num();
    // printf("welcome to GFG from thread = %d\n", tid);
    int randValue, gridPoint;
    for (int i = 1; i < 10000; i++) {
      randValue = rand() % 100 + 1;
      gridPoint = rand() % sizeof(grid);
      if (randValue <= 2) {
        grid[gridPoint] = newline;
      } else if (randValue <= 10) {
        grid[gridPoint] = star;
      } else {
        grid[gridPoint] = space;
      }
    }
    nthreads = omp_get_num_threads();
  }
  printf("%c", '\n');
  grid[999] = '\0';
  printf("%s", grid);

  printf("\nnumber of threads = %d\n", nthreads);
}