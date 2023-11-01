#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RLogger.hxx>
#include "TCanvas.h"

#include <benchmark/benchmark.h>

// this increases RDF's verbosity level as long as the `verbosity` variable is in scope
// auto verbosity =
//    ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(), ROOT::Experimental::ELogLevel::kInfo);

int main(int argc, char **argv)
{
   size_t bulkSize = -1;
   int nbins = -1;
   unsigned long long nvals = -1;
   bool verbose = false;
   char *file;

   int c;
   while ((c = getopt(argc, argv, "b:h:f:v")) != -1) {
      switch (c) {
      case 'b': bulkSize = std::stoul(optarg); break;
      case 'h': nbins = atoi(optarg); break;
      case 'f': file = optarg; break;
   case 'v': verbose = true; break;
      default: std::cout << "Ignoring unknown parse returns: " << char(c) << std::endl;
      }
   }

   if (argc < 3) {
      printf("The parameters are not set!!!\n");
      return 1;
   }

   TCanvas *cv;
   if (verbose) {
      printf("Bulksize: %lu Bins: %d File: %s\n", bulkSize, nbins, file);
      cv = new TCanvas("c", "", 200, 10, 700, 500);
   }

   ROOT::GetROOT();

   auto pageSource = ROOT::Experimental::Detail::RPageSource::Create("Data", file);
   auto df = ROOT::RDataFrame(std::make_unique<ROOT::Experimental::RNTupleDS>(std::move(pageSource)), {}, bulkSize);
   auto mdl = ROOT::RDF::TH1DModel("h1", "h1", nbins, 0, 1);

   auto start = Clock::now();
   auto h1 = df.Histo1D<double>(mdl, "Doubles");
   h1.GetValue();
   auto end = Clock::now();

   if (verbose) {
      h1->Draw();
      cv->SaveAs("test.png");
      // free(cv);
   }

#if not defined(TIME_FILL) && not defined(TIME_STATS) && not defined (TIME_FINDBIN)
   printf("%f\n", std::chrono::duration_cast<fsecs>(end - start).count());
#endif

   return 0;
}
