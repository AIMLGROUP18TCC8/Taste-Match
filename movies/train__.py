import pandas as pd # Import the pandas library for data manipulation

from pathlib import Path
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / 'data.csv')

#df = pd.read_csv('data.csv') # Load the 'data.csv' file into a pandas DataFrame named df
df  # Display the DataFrame to see its initial structure and content
df = df.drop(columns=["Director", "Stars", "Description","Certification", "Unnamed: 0"]) # Drop columns that are not relevant for clustering or have too many missing values
df.isnull().sum() # Check for the number of missing values in each column after dropping
df # Display the DataFrame again after dropping columns
df.duplicated().sum() # Calculate the number of duplicate rows in the DataFrame
df.dtypes # Display the data types of each column in the DataFrame
df["MetaScore"] = df["MetaScore"].fillna(df["MetaScore"].mean()) # Fill missing 'MetaScore' values with the mean of the 'MetaScore' column
df["Gross"] = df["Gross"].fillna(df["Gross"].mean()) # Fill missing 'Gross' values with the mean of the 'Gross' column
df.isnull().sum() # Verify that there are no remaining missing values after imputation
X = df[["Genre"]].copy() # Use only the Genre column as X; exclude all metadata and other features from clustering
X.head() # Display the first few rows of X to inspect the data
type(X["Genre"].iloc[0]) # Check the data type of the first element in the 'Genre' column
import ast # Import the ast module to safely evaluate string representations of Python literals

X["Genre"] = X["Genre"].apply(ast.literal_eval) # Convert the string representation of lists in the 'Genre' column into actual lists
type(X["Genre"].iloc[0]) # Verify that the data type of the first element in the 'Genre' column is now a list
print(X["Genre"].iloc[1]) # Print the second element (at index 1) of the 'Genre' column
print(repr(X["Genre"].iloc[1])) # Print the representation of the second element of the 'Genre' column
print(type(X["Genre"].iloc[1])) # Print the data type of the second element of the 'Genre' column
X["Genre"] = X["Genre"].apply(
    lambda genres: [genre.strip() for genre in genres] # Strip whitespace infront or behind from each genre string within the lists in the 'Genre' column
)
genre_dummies = pd.get_dummies(X["Genre"].explode(),dtype= int ).groupby(level=0).max() # One-hot encode the 'Genre' column:
                                                                                          # - explode() separates genres into individual rows
                                                                                          # - get_dummies() converts genres into binary columns
                                                                                          # - groupby(level=0).max() consolidates back to original index, marking 1 if a genre is present
genre_dummies.head() # Display the first few rows of the created genre_dummies DataFrame
X = X.drop(columns="Genre") # Drop the original 'Genre' column from X as it has been converted to dummy variables

X = pd.concat([X, genre_dummies], axis=1) # Concatenate(join) the original X DataFrame (without 'Genre') with the new genre_dummies DataFrame, axis=1 to represet columns
X.head() # Display the first five rows of the updated X DataFrame
X.shape # Display the dimensions (rows, columns) of the X DataFrame
df.shape # Display the dimensions (rows, columns) of the original df DataFrame for comparison
X.head() # Display the first five rows of X to confirm the changes
from sklearn.preprocessing import StandardScaler # Import StandardScaler for feature scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X) # Scale the features in X using standardization (mean=0, variance=1) and store the result in X_scaled
X_scaled
from sklearn.cluster import KMeans # Import KMeans for clustering

kmeans = KMeans(n_clusters=14, random_state=42, n_init='auto') # Initialize KMeans with 8 clusters (chosen from the elbow method) and a fixed random state, using n_init='auto'
labels = kmeans.fit_predict(X_scaled) # Fit the KMeans model to the scaled genre data and predict cluster labels for each movie

df["Cluster"] = labels  # Add the assigned cluster labels as a new column 'Cluster' to the original DataFrame df, the numbering to start from 1
genre_dummies["Cluster"] = labels  # Add the assigned cluster labels to the genre_dummies DataFrame, starting from 1
genre_dummies.groupby("Cluster").mean(numeric_only=True) # Group the genre_dummies DataFrame by cluster and calculate the mean for each genre, showing the prominence of genres within each cluster
cluster_means = genre_dummies.groupby("Cluster").mean(numeric_only=True) # Group the genre_dummies DataFrame by cluster and calculate the mean for each genre, showing the prominence of genres within each cluster
for cluster in cluster_means.index:
    print(f"\nCluster {cluster}")
    print(cluster_means.loc[cluster].sort_values(ascending=False).head(5))
cluster_names = {
    0: "The Feel-Good Story Seeker (You love heartwarming stories that bring people together.)",
    1: "The Plot-Twist Addict (You love suspense, unexpected twists, and stories that keep you guessing.)",
    2: "The Imagination Explorer (You love stepping into imaginative worlds where anything can happen.)",
    3: "The Hopeless Romantic (You believe in butterflies, big feelings, and love stories worth remembering.)",
    4: "The Real-Life Story Lover (You're drawn to remarkable lives, true stories, and people who make history.)",
    5: "The Wild West Wanderer (You appreciate rugged adventures, frontier legends, and old-school showdowns.)",
    6: "The Crime Story Detective (You love complex cases, morally complicated characters, and gritty stories.)",
    7: "The Fear Chaser (You enjoy the adrenaline rush of a good scare and the mystery behind it.)",
    8: "The Underdog Cheerleader (You love determination, comeback stories, and watching people beat the odds.)",
    9: "The Adventure Seeker (You want big journeys, daring escapes, and stories that take you somewhere new.)",
    10: "The Dark Classic Enthusiast (You appreciate shadowy mysteries, complicated motives, and classic detective stories.)",
    11: "The Mystery Unraveller (You love connecting clues, questioning every detail, and solving the puzzle before the reveal.)",
    12: "The Musical Dreamer (You love stories that turn emotions into music, movement, and unforgettable performances.)",
    13: "The Heart-and-Humor Lover (You enjoy stories that make you laugh, feel deeply, and see a little of yourself on screen.)"
}
df["Cluster_Name"] = df["Cluster"].map(cluster_names) # Create a new column 'Cluster_Name' in df by mapping the numerical cluster IDs to their descriptive names
print(df.columns.tolist()) # Print a list of all column names in the df DataFrame
df.sort_values(by=['Votes', 'Movie Rating'], ascending=[False, False]).head(100) # Sort the entire DataFrame by 'Votes' (descending) and then 'Movie Rating' (descending), and display the top 100 movies with all columns.
genre_counts = genre_dummies.drop(columns=["Cluster"]).sum().sort_values(ascending=False)
print(genre_counts)





import joblib
from pathlib import Path

model_path = Path(__file__).parent / "movie_model.pkl"


joblib.dump({
    "kmeans": kmeans,
    "scaler": scaler,
    "features": X.columns.tolist(),
    "cluster_names": cluster_names,
    "genre_columns": genre_dummies.drop(columns=["Cluster"], errors="ignore").columns.tolist()
}, Path(__file__).parent / "movie_model.pkl")

df.to_csv(Path(__file__).parent / "clustered_movies.csv", index=False)

print("Movie model and clustered movies saved successfully!")