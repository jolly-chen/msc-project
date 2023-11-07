
// NOTE: The RNTuple classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

// Until C++ runtime modules are universally used, we explicitly load the ntuple
// library.  Otherwise triggering autoloading from the use of templated types
// would require an exhaustive enumeration of "all" template instances in the
// LinkDef file.
// R__LOAD_LIBRARY(ROOTNTuple)

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TROOT.h>
#include <TString.h>
#include "TRandom3.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

// Import classes from experimental namespace for the time being
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

std::string kNTupleFileName;

void Generate(size_t n, std::string mode, char *argv[])
{
   // We create a unique pointer to an empty data model
   auto model = RNTupleModel::Create();

   // To define the data model, we create fields with a given C++ type and name.
   // Fields are roughly TTree branches. MakeField returns a shared pointer to a
   // memory location that we can populate to fill the ntuple with data
   auto fldVals = model->MakeField<double>("Doubles");

   // We hand-over the data model to a newly created ntuple of name "Staff",
   // stored in kNTupleFileName In return, we get a unique pointer to an ntuple
   // that we can fill
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "Data", kNTupleFileName);

   TRandom3 r(0);
   if (mode == "uniform") {
      for (size_t i = 0; i < n; i++) {
         *fldVals = r.Rndm();
         ntuple->Fill();
      }
   } else if (mode == "normal") {
      for (size_t i = 0; i < n; i++) {
         *fldVals = r.Gaus(0.5, 0.5);
         ntuple->Fill();
      }
   } else if (mode == "constant") {
      for (size_t i = 0; i < n; i++) {
         *fldVals = 0.5;
         ntuple->Fill();
      }
   } else if (mode == "strided") {
      auto stride = atol(argv[0]);
      auto nbins = atol(argv[1]);

      // double val = 0;
      // for (size_t i = 0; i < n; i++) {
      //    *fldVals = val;
      //    ntuple->Fill();
      //    val += (1/nbins * stride) %
      // }
   } else {
      printf("Mode %s unknown!\n", mode.c_str());
   }
}

void Analyze()
{
   // Get a unique pointer to an empty RNTuple model
   auto model = RNTupleModel::Create();

   // We only define the fields that are needed for reading
   std::shared_ptr<double> fldVals = model->MakeField<double>("Doubles");

   // Create an ntuple and attach the read model to it
   auto ntuple = RNTupleReader::Open(std::move(model), "Data", kNTupleFileName);

   // Quick overview of the ntuple and list of fields.
   ntuple->PrintInfo();

   std::cout << "The first entry in JSON format:" << std::endl;
   ntuple->Show(0);
   // In a future version of RNTuple, there will be support for ntuple->Scan()

   auto c = new TCanvas("c", "", 200, 10, 700, 500);
   TH1F h("h", "Generated values", 100, 0, 1);
   h.SetFillColor(48);

   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      h.Fill(*fldVals);
   }

   h.Draw();
   c->SaveAs("generated.png");
}

// root macro
void generate_ntuple(size_t n, std::string mode, char *argv[])
{
   std::ostringstream ss;
   ss << "input/doubles_" << mode.c_str() << "_" << n << ".root";
   kNTupleFileName = ss.str();

   printf("Generating %s...\n", kNTupleFileName.c_str());

   Generate(n, mode, argv);
   Analyze();
}

int main(int argc, char *argv[])
{
   size_t n = 1e6;
   std::string mode = "uniform";

   int c;
   while ((c = getopt(argc, argv, "n:m:")) != -1) {
      switch (c) {
      case 'n': n = atol(optarg); break;
      case 'm': mode = optarg; break;
      default: std::cout << "Ignoring unknown parse returns: " << char(c) << std::endl;
      }
   }

   generate_ntuple(n, mode, argc > 5 ? &argv[5] : NULL);

   return 0;
}
