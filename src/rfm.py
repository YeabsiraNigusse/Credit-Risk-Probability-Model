# src/rfm.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_rfm(df, snapshot_date):
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

# etc...

def scale_rfm(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled, scaler

def perform_kmeans(rfm_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters, kmeans

def label_high_risk_cluster(rfm, cluster_col='Cluster'):
    cluster_profile = rfm.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_profile['Recency'].idxmax()
    rfm['is_high_risk'] = (rfm[cluster_col] == high_risk_cluster).astype(int)
    return rfm, cluster_profile, high_risk_cluster
