#include <algorithm>
#include <cstdio>
#include "pinMarker.h"


/// Binary search taken from ROOT
template <typename T>
inline long long BinarySearch(long long n, const T *array, T value)
{
   const T *pind;
   pind = std::lower_bound(array, array + n, value);
   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);
}

int main()
{
   int nbins = 16777216;
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i * 1. / nbins);
   auto a_binedges = binedges.data();

   double val = 1;

   _magic_pin_start();
   int bin = BinarySearch(nbins, a_binedges, val);
   _magic_pin_stop();

   printf("bin: %d\n", bin);

   return 0;
}
