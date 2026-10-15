
import pandas as pd
import numpy as np
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from pathlib import Path
df = pd.read_csv(Path(__file__).parent / 'spotify-tracks-dataset.csv')
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')


df = pd.read_csv(Path(__file__).parent / 'spotify-tracks-dataset.csv')
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

# Drop completely identical rows
df = df.drop_duplicates().copy()

# Drop rows with repeated track IDs
df = df.drop_duplicates(subset=['track_id']).copy()

df.shape
text_cols = ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']
numeric_cols = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
]
df[numeric_cols]
df['explicit'] = df['explicit'].fillna(False)
df['explicit'] = df['explicit'].astype(int)
df.isnull().sum()
total_null_rows = df.isnull().any(axis=1).sum()
print(f'Total rows with at least one null: {total_null_rows}')
print(df.isnull().sum()[df.isnull().sum() > 0])
duplicate_track_rows = df.duplicated(subset=['track_id']).sum()
print(f'Duplicate Track Rows: {duplicate_track_rows}')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df[text_cols] = df[text_cols].fillna('Unknown')
# Creating a new feature: duration in minutes

df['duration_min'] = df['duration_ms'] / 60000

# Display the new feature
df[['duration_ms', 'duration_min']].head()
# Encoding Categorical Values

# Convert explicit from True/False to 1/0
df['explicit'] = df['explicit'].astype(int)

df[['explicit']].head()

import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Features used to group music
features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

# Create a folder for saved model files
Path("models").mkdir(exist_ok=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Train K-Means using scaled features
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans.fit_predict(X_scaled)

# Name the clusters using your existing labels
cluster_names = {
    0: "Chill / Acoustic",
    1: "High-Energy / Rock & Workout",
    2: "Upbeat / Dance & Pop",
    3: "Deep Focus / Ambient",
    4: "Moody / Melancholic"
}

df['cluster_name'] = df['cluster'].map(cluster_names)

# Save the fitted model
joblib.dump(kmeans, "models/music_kmeans.pkl")

# Save the fitted scaler
joblib.dump(scaler, "models/music_scaler.pkl")

# Save the exact feature order
joblib.dump(features, "models/music_features.pkl")

# Save the cluster names
joblib.dump(cluster_names, "models/music_cluster_names.pkl")

# Save the cleaned dataset with cluster assignments
df.to_csv("models/clustered_music.csv", index=False)

print("Music model and supporting files saved successfully!")
print(df['cluster_name'].value_counts())
