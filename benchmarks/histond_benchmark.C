#include "ROOT/RDataFrame.hxx"
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleDS.hxx>
#include "TCanvas.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#ifdef LIKWID
#include <likwid-marker.h>
#endif

#ifdef RDF_LOG
#include <ROOT/RLogger.hxx>
#include "ROOT/RDF/Utils.hxx"
// this increases RDF's verbosity level as long as the `verbosity` variable is in scope
auto verbosity =
   ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(), ROOT::Experimental::ELogLevel::kInfo);
#endif

int main(int argc, char **argv)
{
   size_t bulkSize = -1;
   int nbins = -1;
   bool verbose = false;
   bool edges = false;
   char *file;

   int c;
   while ((c = getopt(argc, argv, "b:h:f:ve")) != -1) {
      switch (c) {
      case 'b': bulkSize = std::stoul(optarg); break;
      case 'h': nbins = atoi(optarg); break;
      case 'f': file = optarg; break;
      case 'v': verbose = true; break;
      case 'e': edges = true; break;
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

   ROOT::RDF::TH1DModel mdl;
   if (edges) {
      std::vector<double> e(nbins + 1);
      for (int i = 0; i <= nbins; i++)
         e[i] = i * 1. / nbins;

      if (verbose) {
         printf("Edges:\n");
         for (size_t i = 0; i < e.size(); i++) {
            printf("%f ", e[i]);
         }
         printf("\n");
      }
      mdl = ROOT::RDF::TH1DModel("h1", "h1", nbins, e.data());
   } else {
      mdl = ROOT::RDF::TH1DModel("h1", "h1", nbins, 0, 1);
   }

#ifdef LIKWID
   LIKWID_MARKER_REGISTER("mfilln");
   LIKWID_MARKER_INIT;
#endif

   /// Benchmark START
   auto start = Clock::now();
   auto h1 = df.Histo1D<double>(mdl, "Doubles");
   auto &result = h1.GetValue();
   auto end = Clock::now();
   /// Benchmark EbND

#ifdef LIKWID
   LIKWID_MARKER_CLOSE;
#endif

   if (verbose) {
      h1->Draw();
      cv->SaveAs("test.png");
   }

   // To check the result
   std::ostringstream os;
   os << std::filesystem::path(file).stem().string() << "_h" << nbins << "_e" << (edges ? "1" : "0") << ".out";

   std::ofstream expected;
   expected.open(os.str());

   auto histArray = result.GetArray();
   for (int i = 0; i < nbins + 2; i++) {
      expected << histArray[i] << " ";
   }
   expected << "\n";

   double stats[4];
   result.GetStats(stats);
   for (int i = 0; i < 4; i++) {
      expected << stats[i] << " ";
   }
   expected << "\n" << result.GetEntries() << "\n";

   // Report timing
   printf("findbin:%f\n", result.tfindbin);
   printf("fill:%f\n", result.tfill);
   printf("stats:%f\n", result.tusb);
   printf("total:%f\n", std::chrono::duration_cast<fsecs>(end - start).count());

   return 0;
}
