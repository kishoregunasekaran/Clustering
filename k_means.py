import numpy as np

from sklearn.cluster import KMeans








data = np.array([[1,1], [2,1], [2,3], [3,2], [4,3], [5,5]])







# Manually initialize centroids







initial_centroids = np.array([[2, 1], [2, 3]])







# Run KMeans with manually initialized centroids







kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, random_state=0).fit(data)







# Get labels and centers







labels = kmeans.labels_







centers = kmeans.cluster_centers_







cluster_1 = data[labels == 0]







cluster_2 = data[labels == 1]







# Print results in the desired format







print(f"C1 : {cluster_1.tolist()}")







print(f"C2 : {cluster_2.tolist()}")







print(f"Cluster centers : {centers}")

























