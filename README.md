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
pileup - energy density from other p-p collisions other than primary(triggered) collision 
JVT = jet vertex tagger - fraction of jet pt comming from primary vertex 
JVF = jet vertex fraction

- **jets_ConeTruthLabelID** - truth label, not from data
- **jets_HadronConeExclExtendedTruthLabelID** - truth label, not from data
- **jets_HadronConeExclTruthLabelID** - truth label, not from data
- **jets_ActiveArea4vec_(eta/m/phi/pt)** - energy carried by pileup in the cross section of the jet
- **jets_DetectorEta** - uncorrected eta of jet (not important)
- **jets_EMFrac** - fraction of energy of a hadron deposited in electromagnetic calorimeter (decay in hadron calorimeter)
- **jets_FracSamplingMax** - fraction of energy deposited in the calorimeter cell with the highest energy deposition
- **jets_FracSamplingMaxIndex** - index of the calorimeter cell with the highest energy deposition
- **jets_GhostMuonSegmentCount** - number of muon segments (muon spectrometer) hit by segments of the jet which did not decay in the hadron calorimeter
- **jets_JetConstitScaleMomentum_(eta/m/phi/pt)** - total energie of jet constituents (not important)
- **jets_chf** - charged fraction
- **jets_(eta/m/phi/pt)** - reconstructed jet kinematics
- **jets_Width** - weighted (pt) average distance of constituents from axis oj jet
- **jets_Jvt** - fraction of jet pt comming from primary vertex 
- **jets_Timing** - jet time with respect to the collsion time
- **jets_JVFCorr**
- **jets_JvtRpt**
- **jets_fJVT**
- **jets_passFJVT**
- **jets_passJVT**

### Branched variables (branched at verticies)
- **jets_ChargedPFOWidthPt1000** - charged PFOs width with pt > 1000
- **jets_TrackWidthPt1000** - track width with pt > 1000
- **jets_NumChargedPFOPt1000** - number of charged PFOs with pt > 1000
- **jets_NumChargedPFOPt500** - number of charged PFOs with pt > 500
- **jets_SumPtChargedPFOPt500** - sum pt of charged PFOs with pt > 500
- **jets_SumPtTrkPt500** - sum of pt of tracks with pt > 500

- **jets_EnergyPerSampling** - energy of jet deppsited in given calorimeter layer (length of variables is 28)


## Notes

Shuffling matters because we flatten each event to individial jets and the order remains.  
p_T cut na 50 GeV
vhodne cuty na uletene premenne
fyzikalne test spravnosti taggeru (perEvent)



