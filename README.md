# Multi-Omics Integration Dashboard (In Progress)

This project is an interactive bioinformatics dashboard designed to analyze and visualize multi-omics data, specifically integrating transcriptomics (RNA-seq) and proteomics (mass spectrometry) datasets.

⚡ **Note:** This project is currently a work in progress. More features, datasets, and polish will be added soon.

## Features Implemented
- 📥 Upload Gene Expression, Protein Expression, and Metadata CSV files
- 📊 Perform Principal Component Analysis (PCA) for transcriptomics and proteomics separately
- 🔬 Integrate multi-omics layers via Gene-Protein Correlation Heatmaps
- 🎯 Identify important biomarkers using XGBoost feature importance ranking
- 🌐 Simple and interactive dashboard built using Streamlit

## Planned Features
- 🌟 Integration with real public datasets (TCGA + CPTAC)
- 📈 Advanced visualizations (Volcano plots, Correlation Networks)
- 📋 Downloadable biomarker reports
- 🚀 Deployment on Streamlit Cloud for public access

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/multiomics_dashboard.git
   cd multiomics_dashboard
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate       # (Windows)

   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run dashboard/app.py
   ```

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Plotly
- Seaborn

---

Made by Nayan Chaudhari
