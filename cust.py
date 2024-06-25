import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('customers.csv')

# Display the first few rows of the dataset
print(data.head())

# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Preprocess the data
# Create a column transformer with OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Apply the transformations to the dataset
data_processed = preprocessor.fit_transform(data)

# Convert the processed data back to a DataFrame for easier handling
data_processed_df = pd.DataFrame(data_processed.toarray() if hasattr(data_processed, "toarray") else data_processed)

# Choose the number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data_processed_df)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the Elbow Method graph, choose optimal number of clusters
optimal_clusters = 3

# Fit the K-Means model
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(data_processed_df)

# Add the cluster labels to the original dataframe
data['Cluster'] = clusters

# Visualize the clusters using PCA 
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_processed_df)

# Create a DataFrame with the PCA results
data_pca_df = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'])
data_pca_df['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=data_pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)
plt.title('Customer Segments')
plt.show()

# Group by cluster and calculate the mean for only numerical columns
numerical_data = data[numerical_cols]
numerical_data['Cluster'] = clusters
print(numerical_data.groupby('Cluster').mean())
