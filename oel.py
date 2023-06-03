import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

url = r"C:\Users\shahz\Desktop\oel\cars_clus.csv"
df = pd.read_csv(url)

df = df.dropna()
df = df.replace('$null$', np.nan)
df = df.dropna()
df = df.drop(['manufact', 'model'], axis=1)
df['type'] = df['type'].replace({'sedan': 1, 'sports': 2, 'wagon': 3, 'coupe': 4, 'hatchback': 5})

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

cleaned_data_file = r"C:\Users\shahz\Desktop\oel\cleaned_data.csv"
df.to_csv(cleaned_data_file, index=False)

print(df.describe())

correlation_matrix = df.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init=10)  # Explicitly set n_init=10
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10)  # Explicitly set n_init=10
kmeans.fit(df_scaled)

silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

df['cluster'] = kmeans.labels_

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'] = df['price'].round(2)

# Save cluster data into separate CSV files
for cluster in range(n_clusters):
    cluster_vehicles = df[df['cluster'] == cluster]
    cluster_filename = fr"C:\Users\shahz\Desktop\oel\cluster_{cluster}.csv"
    cluster_vehicles.to_csv(cluster_filename, index=False)

plt.scatter(df['mpg'], df['price'], c=df['cluster'])
plt.xlabel('MPG')
plt.ylabel('Price')
plt.title('Vehicle Clusters')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for cluster in range(n_clusters):
    cluster_vehicles = df[df['cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_vehicles)
    print()
