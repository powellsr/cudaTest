#ifndef _CPUADD_CPP_
#define _CPUADD_CPP_

#include <iostream>
#include "cpuadd.h"

void cpuadd(double *a, double *b, double*c, int n) {
  std::cout << "...CPU add, using loop to add" << std::endl;
  for (size_t id = 0; id < n; ++id) 
    c[id] = a[id] + b[id];
}

#endif
