#           LightEdge-IDS - A Lightweight Real-Time Intrusion Detection System for Cyber Physical Attacks in Water Distribution Systems

## Problem Statement
"To detect real-time cyber-physical attacks in water distribution systems using a lightweight, zone-adaptive Intrusion Detection System based on Graph Neural Networks and neural adapters."

## Objectives

1. Model water networks using graph structures.
2. Learn spatio-temporal patterns using GNN-GRU architecture.
3. Enable zone-wise anomaly detection using adapters.
4. Detect cyber-physical attacks in water networks.
5. Develop lightweight real-time intrusion detection system.

## System Architecture

The LightEdge-IDS system follows a multi-stage pipeline:
1. Graph Construction  
   - Water network parsed from EPANET (.inp file)
   - Graph built using NetworkX
2. Subgraph Construction  
   - Sensor nodes mapped to network
   - Condensed graph created using shortest paths
3. Zone Division  
   - Leiden community detection used to divide graph into zones
4. Zone-wise Modeling  
   - Each zone processed using adapter models
5. Global Modeling  
   - Zone embeddings combined using GNN + GRU
6. Prediction  
   - Final anomaly detection using model outputs
  
## Dataset

The system uses the BATADAL dataset, which contains:
- Sensor readings from a water distribution network
- Time-series data (flow, pressure, etc.)
- Attack labels indicating cyber-physical attacks
  
Key Details:
- Timestamp column: DATETIME
- Label column: ATT_FLAG / ATTACK_FLAG
- Multiple sensor nodes mapped to graph nodes
  
Preprocessing includes:
- Missing value imputation
- Resampling and interpolation
- Feature normalization

## Project Structure

├───inputs
│   ├───combined
│   └───test_data
├───outputs
│   ├───analysis_results
│   ├───IDS
│   │   ├───data
│   │   └───outputs
│   │       ├───embeddings
│   │       ├───scalers
│   │       └───ts
│   ├───models
│   ├───subgraph_constr
│   └───zone_div
└───src
    └───__pycache__

## Installation

1. Clone the repository
   
git clone https://github.com/srmsminiproject/LightEdge-IDS.git
cd LightEdge-IDS

2. Install dependencies
pip install -r requirements.txt

##  Usage
cd src
### Preprocessing and Subgraph Construction
  -python 01_Selection_Graph_Construction.py
  -Results stored in ../outputs/subgraph_constr

### Zone Division
  -python 02_zone_division
  -Results stored in ../outputs/zone_div
  
### Train the Models and Evaluation
  -python src/03_driver.py
  -Models stored in ../outputs/models
  -Evaluation results stored in ../outputs/analysis_results

### Run UI Demonstration

streamlit run demonstration.py


## Technologies Used

* Python
* PyTorch / TensorFlow
* Scikit-learn
* NumPy
* Pandas


## Future Work

*  Deployment on Raspberry Pi
*  Integration with real-time dashboard
*  Further optimization for ultra-low latency
*  Expansion to multi-device network monitoring

##  Contributors

-Guide: Dr.Sunil K S
-Amritha Remesh
-Veena Roy
-Vidya Roy
-Vivek D V

