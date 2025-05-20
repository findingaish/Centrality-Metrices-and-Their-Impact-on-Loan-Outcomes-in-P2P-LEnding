Centrality Effects and Their Impact on Loan Outcomes in Peer-to-Peer Lending
==================================================

This repository contains the source code, dataset, and output results for the project 
"Network Centrality Effects on Peer-to-Peer Lending". The project analyzes the impact of 
network centrality measures on loan outcomes using machine learning models.

Folder Structure
----------------

- src/: Contains two Jupyter Notebook files for processing and analysis:
  - P2Plending.ipynb: Generates the Minimum Spanning Tree (MST) using borrower attributes 
    and computes centrality measures (degree, betweenness, PageRank, etc.).
  - p2p_evaluation_models.ipynb: Trains and evaluates machine learning models (Logistic Regression, 
    Random Forest, XGBoost) using borrower attributes and centrality measures.
  
- dataset/: Contains the input data file new_file_5000.csv with records for 5000 borrowers.

- output/: Stores the output images (plots, visualizations, and analysis results).

Steps to Execute the Project
----------------------------

1. Set up the environment:
   - Ensure Python is installed on your system (version 3.7 or higher).
   - Install the required dependencies:
     pip install -r requirements.txt

2. Prepare the dataset:
   - Place the dataset file (new_data_5000.csv) in the dataset/ folder.

3. Run MST calculation:
   - Open the src/P2Plending.ipynb file in Jupyter Notebook or any compatible environment.
   - Execute the notebook cells step-by-step to:
     - Preprocess the dataset.
     - Compute Gower's and Euclidean distances.
     - Construct MSTs and compute network centrality metrics.
   - The generated outputs (plots, metrics) will be saved in the output/ folder.

4. Train and evaluate ML models:
   - Open the src/p2p_evaluation_models.ipynb file.
   - Execute the notebook cells to:
     - Load borrower attributes and centrality metrics.
     - Train Logistic Regression, Random Forest, and XGBoost classifiers.
     - Evaluate models using metrics like accuracy, precision, recall, F1-score, and AUC.
   - Comparison results and visualizations will be saved in the output/ folder.

5. Review outputs:
   - Access the output/ folder to view saved images and analysis results.

Dependencies
------------

Ensure the following Python libraries are installed:
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- networkx
- shap

For easier installation, run:
pip install numpy pandas scikit-learn xgboost matplotlib seaborn networkx shap

Project Workflow Overview
-------------------------

1. Data Preparation:
   - Cleaned and preprocessed the dataset.
   - Calculated Gower and Euclidean distance matrices.

2. Network Centrality:
   - Constructed MSTs using distance matrices.
   - Computed centrality metrics (Degree, Degree centrality, Betweenness, PageRank, Strength).

3. Machine Learning:
   - Used centrality metrics and borrower attributes as features.
   - Trained three classifiers (Logistic Regression, Random Forest, XGBoost).
   - Compared models using SHAP for feature importance analysis.

4. Results Visualization:
   - Visualized key findings and model performances through saved plots.

