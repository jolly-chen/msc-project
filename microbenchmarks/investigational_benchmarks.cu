#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <new>
#include <thrust/binary_search.h>

using Clock = std::chrono::steady_clock;
using fsecs = std::chrono::duration<double>;

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      fprintf(stderr, (func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}


/**
 * Returns aligned pointers when allocations are requested. Default alignment
 * is 64B = 512b, sufficient for AVX-512 and most cache line sizes.
 * Taken from: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
 *
 * @tparam ALIGNMENT_IN_BYTES Must be a positive power of 2.
 */
template <typename ElementType, std::size_t ALIGNMENT_IN_BYTES = 64>
class AlignedAllocator {
private:
   static_assert(ALIGNMENT_IN_BYTES >= alignof(ElementType),
                 "Beware that types like int have minimum alignment requirements "
                 "or access will result in crashes.");

public:
   using value_type = ElementType;
   static std::align_val_t constexpr ALIGNMENT{ALIGNMENT_IN_BYTES};

   /**
    * This is only necessary because AlignedAllocator has a second template
    * argument for the alignment that will make the default
    * std::allocator_traits implementation fail during compilation.
    * @see https://stackoverflow.com/a/48062758/2191065
    */
   template <class OtherElementType>
   struct rebind {
      using other = AlignedAllocator<OtherElementType, ALIGNMENT_IN_BYTES>;
   };

public:
   constexpr AlignedAllocator() noexcept = default;

   constexpr AlignedAllocator(const AlignedAllocator &) noexcept = default;

   template <typename U>
   constexpr AlignedAllocator(AlignedAllocator<U, ALIGNMENT_IN_BYTES> const &) noexcept
   {
   }

   [[nodiscard]] ElementType *allocate(std::size_t nElementsToAllocate)
   {
      if (nElementsToAllocate > std::numeric_limits<std::size_t>::max() / sizeof(ElementType)) {
         throw std::bad_array_new_length();
      }

      auto const nBytesToAllocate = nElementsToAllocate * sizeof(ElementType);
      return reinterpret_cast<ElementType *>(::operator new[](nBytesToAllocate, ALIGNMENT));
   }

   void deallocate(ElementType *allocatedPointer, [[maybe_unused]] std::size_t nBytesAllocated)
   {
      /* According to the C++20 draft n4868 § 17.6.3.3, the delete operator
       * must be called with the same alignment argument as the new expression.
       * The size argument can be omitted but if present must also be equal to
       * the one used in new. */
      ::operator delete[](allocatedPointer, ALIGNMENT);
   }
};
template<typename T, std::size_t ALIGNMENT_IN_BYTES = 64>
using AlignedVector = std::vector<T, AlignedAllocator<T, ALIGNMENT_IN_BYTES> >;


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

template <typename T>
__global__ void BinarySearchGPU(size_t n, const T *array, const T *vals)
{
   const T *pind;
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

   pind = thrust::lower_bound(thrust::seq, array, array + n, vals[tid]);
}


// template <typename T>
// __device__ size_t BinarySearchGPU(size_t n, const T *array, T value)
// {
//    const T *pind;

//    pind = thrust::lower_bound(thrust::seq, array, array + n, value);
//    if ((pind != array + n) && (*pind == value))
//       return (pind - array);
//    else
//       return (pind - array - 1);
// }

// template<typename T>
// __global__ void BinarySearchKernel(size_t n, size_t size, const t *array, T value)
// {
//    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
//    unsigned int localTid = threadIdx.x;
//    unsigned int stride = blockDim.x * gridDim.x; // total number of threads

//    // Fill local histogram
//    for (auto i = tid; i < n; i += stride) {
//       auto r = CUDAHelpers::BinarySearch(size, array, value);
//    }
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void BM_PauseOverhead(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        AlignedVector<double, 64> v(1e7);
        for (size_t i = 0; i < v.size(); i++) v[i] = rand();
        auto data = v.data();           // Allow v.data() to be clobbered. Pass as non-const
        benchmark::DoNotOptimize(data); // lvalue to avoid undesired compiler optimizations
        benchmark::ClobberMemory();     // Force data to be written to memory.

        state.ResumeTiming();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
BENCHMARK(BM_PauseOverhead)->Unit(benchmark::kSecond);

static void BM_ManualOverhead(benchmark::State &state) {
    for (auto _ : state) {
        AlignedVector<double, 64> v(1e7);
        for (size_t i = 0; i < v.size(); i++) v[i] = rand();
        auto data = v.data();           // Allow v.data() to be clobbered. Pass as non-const
        benchmark::DoNotOptimize(data); // lvalue to avoid undesired compiler optimizations
        benchmark::ClobberMemory();     // Force data to be written to memory.

        auto start = Clock::now();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto end = Clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_ManualOverhead)->UseManualTime()->Unit(benchmark::kSecond);

template <typename T>
inline long long LinearSearch(T *arr, long long n, T val) {
    for (auto i = 0; i < n; i++) {
        if (arr[i] > val) {
            return i-1;
        }
    }
    return -1;
}

// prun -np 1 -v  likwid-pin -c C0:0  ./gbenchmark_evaluation --benchmark_filter=Linear --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6-cpu/gbenchmark_evaluation.json
// prun -np 1 -v  likwid-pin -c C0:0  ./gbenchmark_evaluation --benchmark_filteLinear --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true  &>> out
static void BM_LinearSearch(benchmark::State &state) {
    long long bin;
    long long repetitions = 1000;
    int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
    double val = nbins * (state.range(1) / 4.);

    AlignedVector<double, 64> binedges(nbins);
    auto data = binedges.data();
    for (auto i = 0; i < nbins; i++) binedges[i] = i;

    for (auto _ : state) {
        for (int n = 0; n < repetitions; n++) {
            bin = LinearSearch(data, nbins, val);
        }
    }

    state.counters["repetitions"] = repetitions;
    state.counters["nbins"] = nbins;
    state.counters["val"] = val;
    state.counters["bin"] = bin;
}
BENCHMARK(BM_LinearSearch)
    ->ArgsProduct({benchmark::CreateRange(8, 33554432, /*multi=*/8),
                   {0, 1, 2, 3, 4}})
    ->Unit(benchmark::kMillisecond);


// prun -np 1 -v likwid-pin -c  C0:0 ./cpu-microbenchmarks --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6-cpu/cpu_microbenchmarks.json
// static void BM_BinarySearch(benchmark::State &state) {
//    long long bin;
//    long long repetitions = 1000000;
//    int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
//    // double val = nbins; // Last element
//    double val = nbins * (state.range(1) / 4.);

//    AlignedVector<double, 64> binedges(nbins);
//    auto data = binedges.data();
//    for (auto i = 0; i < nbins; i++) binedges[i] = i;

//    for (auto _ : state) {
//       for (int n = 0; n < repetitions; n++) {
//          bin = BinarySearch(nbins, data, val);
//       }
//    }

//    state.counters["repetitions"] = repetitions;
//    state.counters["nbins"] = nbins;
//    state.counters["val"] = val;
//    state.counters["bin"] = bin;
// }
// BENCHMARK(BM_BinarySearch)
//    ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2),
//                   {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
//    ->Unit(benchmark::kMillisecond);


// Nullify weights of under/overflow bins to exclude them from stats
template <unsigned int Dim, unsigned int BlockSize>
__global__ void ExcludeUOverflowKernel(int *bins, double *weights, int *nBinsAxis, size_t bulkSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   for (auto i = tid; i < bulkSize * Dim; i += stride) {
      if (bins[i] <= 0 || bins[i] >= nBinsAxis[i / bulkSize] - 1) {
         weights[i % bulkSize] = 0.;
      }
   }
}

// prun -np 1 -v likwid-pin -c  C0:0 ./cpu-microbenchmarks --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6-cpu/cpu_microbenchmarks.json
// static void BM_BinarySearchConstantGPU(benchmark::State &state) {
//    long long bin;
//    long long repetitions = 1000000;
//    int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
//    double val = nbins * (state.range(1) / 4.);
//    int blocksize = state.range(3);

//    AlignedVector<double, 64> binedges(nbins);
//    auto data = binedges.data();
//    for (auto i = 0; i < nbins; i++) binedges[i] = i;

//    double *d_binedges;
//    ERRCHECK(cudaMalloc((void **)&d_binedges, nbins * sizeof(double)));
//    ERRCHECK(cudaMemcpy(d_binedges, binedges.data(), nbins * sizeof(double), cudaMemcpyHostToDevice));

//    for (auto _ : state) {
//       for (int n = 0; n < repetitions; n++) {
//          BinarySearchGPU<<<1, 1>>>(nbins, d_binedges, val);
//          cudaDeviceSynchronize();
//       }
//    }

//    state.counters["repetitions"] = repetitions;
//    state.counters["nbins"] = nbins;
//    state.counters["val"] = val;
//    // state.counters["numblocks"] = val;
//    // state.counters["blocksize"] = val;
// }
// BENCHMARK(BM_BinarySearchConstantGPU)
//    ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2),
//                   {0, 1, 2, 3, 4},  // Args only accepts integer, so this is a hacky way to get [0, 0.25, 0.5, 0.75, 1]
//                   // benchmark::CreateRange(1, 1024, /*multi=*/2)}
//    })->Unit(benchmark::kMillisecond);

static void BM_BinarySearchConstantGPU(benchmark::State &state) {

   long long repetitions = 1000;
   int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   int blockSize = state.range(2);
   int numBlocks = bulkSize % blockSize == 0 ? bulkSize / blockSize : bulkSize / blockSize + 1;
   double val = nbins * (state.range(3) / 4.);

   AlignedVector<double, 64> vals(bulkSize);
   for (auto i = 0; i < bulkSize; i++) vals[i] = val;
   double *d_vals;
   ERRCHECK(cudaMalloc((void **)&d_vals, bulkSize * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_vals, vals.data(), bulkSize * sizeof(double), cudaMemcpyHostToDevice));

   AlignedVector<double, 64> binedges(nbins);
   auto data = binedges.data();
   for (auto i = 0; i < nbins; i++) binedges[i] = i;
   double *d_binedges;
   ERRCHECK(cudaMalloc((void **)&d_binedges, nbins * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_binedges, binedges.data(), nbins * sizeof(double), cudaMemcpyHostToDevice));

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      for (int n = 0; n < repetitions; n++) {
         cudaEventRecord(start);
         BinarySearchGPU<<<numBlocks, blockSize>>>(nbins, d_binedges, d_vals);
         cudaEventRecord(stop);

         cudaEventSynchronize(stop);
         float elapsed_milliseconds = 0;
         cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
         state.SetIterationTime(elapsed_milliseconds);
      }
   }

   state.counters["repetitions"] = repetitions;
   state.counters["nbins"] = nbins;
   state.counters["val"] = val;
   state.counters["bulksize"] = bulkSize;
   state.counters["numblocks"] = numBlocks;
   state.counters["blocksize"] = blockSize;
   ERRCHECK(cudaFree(d_binedges));
}
BENCHMARK(BM_BinarySearchConstantGPU)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/2), // Bulkszie
                  benchmark::CreateRange(32, 1024, /*multi=*/2), // blocksize
                  {0, 2, 4},  // Args only accepts integer, so this is a hacky way to get [0, 0.25, 0.5, 0.75, 1]
   })
   ->Unit(benchmark::kMillisecond)
   ->UseManualTime();


static void BM_BinarySearchRandomGPU(benchmark::State &state) {
   long long repetitions = 1000;
   long nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   int blockSize = state.range(2);
   int numBlocks = bulkSize % blockSize == 0 ? bulkSize / blockSize : bulkSize / blockSize + 1;

   std::random_device rd;  // a seed source for the random number engine
   std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
   std::uniform_int_distribution<> distrib(0, nbins);
   AlignedVector<double, 64> vals(bulkSize);
   for (auto i = 0; i < bulkSize; i++) vals[i] = distrib(gen);
   double *d_vals;
   ERRCHECK(cudaMalloc((void **)&d_vals, bulkSize * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_vals, vals.data(), bulkSize * sizeof(double), cudaMemcpyHostToDevice));

   AlignedVector<double, 64> binedges(nbins);
   for (auto i = 0; i < nbins; i++) binedges[i] = i;
   double *d_binedges;
   ERRCHECK(cudaMalloc((void **)&d_binedges, nbins * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_binedges, binedges.data(), nbins * sizeof(double), cudaMemcpyHostToDevice));

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      for (int n = 0; n < repetitions; n++) {
         cudaEventRecord(start);
         BinarySearchGPU<<<numBlocks, blockSize>>>(nbins, d_binedges, d_vals);
         cudaEventRecord(stop);

         cudaEventSynchronize(stop);
         float elapsed_milliseconds = 0;
         cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
         state.SetIterationTime(elapsed_milliseconds);
      }
   }

   state.counters["repetitions"] = repetitions;
   state.counters["nbins"] = nbins;
   state.counters["bulksize"] = bulkSize;
   state.counters["numblocks"] = numBlocks;
   state.counters["blocksize"] = blockSize;
   ERRCHECK(cudaFree(d_binedges));
}
BENCHMARK(BM_BinarySearchRandomGPU)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/2), // Bulkszie
                  benchmark::CreateRange(32, 1024, /*multi=*/2), // blocksize
   })
   ->Unit(benchmark::kMillisecond)
   ->UseManualTime();

BENCHMARK_MAIN();

