import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1

st.title('Multi-Omics Integration Dashboard')

if st.session_state.page == 1:
    st.header('1. Upload Your Data')

    # File uploaders
    gene_file = st.file_uploader('Upload Gene Expression CSV', type=['csv'])
    protein_file = st.file_uploader('Upload Protein Expression CSV', type=['csv'])
    meta_file = st.file_uploader('Upload Metadata CSV', type=['csv'])

    if gene_file and protein_file and meta_file:
        # Load data
        gene_df = pd.read_csv(gene_file)
        protein_df = pd.read_csv(protein_file)
        meta_df = pd.read_csv(meta_file)

        st.success('All files uploaded successfully!')

        # Show previews
        st.subheader('Gene Expression Data Preview')
        st.write(gene_df.head())

        st.subheader('Protein Expression Data Preview')
        st.write(protein_df.head())

        st.subheader('Metadata Preview')
        st.write(meta_df.head())

        if st.button('Submit and Proceed'):
            st.session_state.gene_df = gene_df
            st.session_state.protein_df = protein_df
            st.session_state.meta_df = meta_df
            st.session_state.page = 2
            st.rerun()

elif st.session_state.page == 2:
    st.header('2. Principal Component Analysis (PCA)')

    gene_df = st.session_state.gene_df
    protein_df = st.session_state.protein_df
    meta_df = st.session_state.meta_df

    def run_pca(df, label):
        features = df.drop(columns=[df.columns[0]]).T
        pca = PCA(n_components=2)
        components = pca.fit_transform(features)
        comp_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        comp_df['Sample_ID'] = features.index
        merged_df = comp_df.merge(meta_df, on='Sample_ID')
        fig = px.scatter(merged_df, x='PC1', y='PC2', color='Condition', title=f'PCA Plot - {label}', hover_data=['Sample_ID'])
        return fig

    st.subheader('Gene Expression PCA')
    st.plotly_chart(run_pca(gene_df, "Gene Expression"))

    st.subheader('Protein Expression PCA')
    st.plotly_chart(run_pca(protein_df, "Protein Expression"))

    if st.button('Next: Multi-Omics Integration'):
        st.session_state.page = 3
        st.rerun()

elif st.session_state.page == 3:
    st.header('3. Gene-Protein Correlation Heatmap')

    gene_df = st.session_state.gene_df
    protein_df = st.session_state.protein_df

    gene_features = gene_df.drop(columns=[gene_df.columns[0]]).T
    protein_features = protein_df.drop(columns=[protein_df.columns[0]]).T

    if list(gene_features.index) == list(protein_features.index):
        correlation_matrix = pd.DataFrame()

        for gene in gene_df['Gene']:
            gene_vals = gene_df[gene_df['Gene'] == gene].iloc[:, 1:].values.flatten()
            correlations = []
            for protein in protein_df['Protein']:
                protein_vals = protein_df[protein_df['Protein'] == protein].iloc[:, 1:].values.flatten()
                corr = pd.Series(gene_vals).corr(pd.Series(protein_vals))
                correlations.append(corr)
            correlation_matrix[gene] = correlations
        
        correlation_matrix.index = protein_df['Protein']

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.error('Sample IDs do not match between gene and protein data!')

elif st.session_state.page == 4:
    st.header('4. Biomarker Discovery using XGBoost')

    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    gene_df = st.session_state.gene_df
    protein_df = st.session_state.protein_df
    meta_df = st.session_state.meta_df

    # Prepare gene data
    gene_features = gene_df.drop(columns=[gene_df.columns[0]]).T
    gene_features['Sample_ID'] = gene_features.index
    merged_gene = gene_features.merge(meta_df, on='Sample_ID')
    X_gene = merged_gene.drop(columns=['Sample_ID', 'Condition'])
    y_gene = merged_gene['Condition']

    # Encode labels (Healthy=0, Disease=1)
    le = LabelEncoder()
    y_gene_encoded = le.fit_transform(y_gene)

    # Train XGBoost for genes
    model_gene = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_gene.fit(X_gene, y_gene_encoded)

    # Feature importance for genes
    gene_importance = model_gene.feature_importances_
    top_gene_idx = gene_importance.argsort()[::-1][:5]
    top_genes = X_gene.columns[top_gene_idx]
    top_gene_importances = gene_importance[top_gene_idx]

    # Prepare protein data
    protein_features = protein_df.drop(columns=[protein_df.columns[0]]).T
    protein_features['Sample_ID'] = protein_features.index
    merged_protein = protein_features.merge(meta_df, on='Sample_ID')
    X_protein = merged_protein.drop(columns=['Sample_ID', 'Condition'])
    y_protein = merged_protein['Condition']
    y_protein_encoded = le.fit_transform(y_protein)

    # Train XGBoost for proteins
    model_protein = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_protein.fit(X_protein, y_protein_encoded)

    # Feature importance for proteins
    protein_importance = model_protein.feature_importances_
    top_protein_idx = protein_importance.argsort()[::-1][:5]
    top_proteins = X_protein.columns[top_protein_idx]
    top_protein_importances = protein_importance[top_protein_idx]

    # Plot Top Genes
    st.subheader('Top 5 Important Genes (XGBoost)')
    fig1 = px.bar(x=top_genes, y=top_gene_importances, labels={'x': 'Gene', 'y': 'Importance'})
    st.plotly_chart(fig1)

    # Plot Top Proteins
    st.subheader('Top 5 Important Proteins (XGBoost)')
    fig2 = px.bar(x=top_proteins, y=top_protein_importances, labels={'x': 'Protein', 'y': 'Importance'})
    st.plotly_chart(fig2)
