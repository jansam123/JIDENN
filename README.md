# Jet Identification Deep Neural Network (JIDENN) 

## Introduction 
This project aims to classify whether a jet came from a quark or gluon. 
Initialy we try to match the results achvieved with BDT using high level variables (derived). 
To do this we implemented few NN models and also BDT to compare the results. 

Afterwards we plan to utilize [transformer architecture](https://arxiv.org/abs/1706.03762) and information about the jet constituents to exceed the BDT results.

## Explanation of jets variables
Particle flow - algorithm to stop double counting of particle tracks and clusters 
PFO = particle flow object
tracks = charged particles measured by the tracker (inner detector)
width - weighted (pt) average distance from axis oj jet


- **jets_ActiveArea4vec_eta**
- **jets_ActiveArea4vec_m**
- **jets_ActiveArea4vec_phi**
- **jets_ActiveArea4vec_pt**
- **jets_ConeTruthLabelID**
- **jets_DetectorEta**
- **jets_EMFrac**
- **jets_FracSamplingMax**
- **jets_FracSamplingMaxIndex**
- **jets_GhostMuonSegmentCount**
- **jets_HadronConeExclExtendedTruthLabelID**
- **jets_HadronConeExclTruthLabelID**
- **jets_JVFCorr**
- **jets_JetConstitScaleMomentum_eta**
- **jets_JetConstitScaleMomentum_m**
- **jets_JetConstitScaleMomentum_phi**
- **jets_JetConstitScaleMomentum_pt**
- **jets_JvtRpt**
- **jets_Width**
- **jets_fJVT**
- **jets_passFJVT**
- **jets_passJVT**
- **jets_Jvt** - fraction of jet pt comming from primary vertex 
- **jets_Timing**
- **jets_chf** - charged fraction
- **jets_eta**
- **jets_m**
- **jets_phi**
- **jets_pt**

### Branched variables (branched at verticies)
- **jets_ChargedPFOWidthPt1000** - charged PFOs width with pt > 1000
- **jets_TrackWidthPt1000** - track width with pt > 1000
- **jets_NumChargedPFOPt1000** - number of charged PFOs with pt > 1000
- **jets_NumChargedPFOPt500** - number of charged PFOs with pt > 500
- **jets_SumPtChargedPFOPt500** - sum pt of charged PFOs with pt > 500
- **jets_SumPtTrkPt500** - sum of pt of tracks with pt > 500

- **jets_EnergyPerSampling** - energy of jet deppsited in given calorimeter layer (length of variables is 28)


## Notes

[ML in ROOT](https://root.cern/manual/tmva/)