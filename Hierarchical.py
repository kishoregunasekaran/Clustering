import numpy as np







import matplotlib.pyplot as plt







from scipy.cluster.hierarchy import linkage, dendrogram







# Given distance matrix (the upper triangle)







distance_matrix = np.array([







[0, 9, 3, 6, 11],







[9, 0, 7, 5, 10],







[3, 7, 0, 9, 2],







[6, 5, 9, 0, 8],







[11, 10, 2, 8, 0]







])







# Convert the distance matrix into a condensed form required by scipy







condensed_distance_matrix = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]







# Perform complete-linkage hierarchical clustering







Z = linkage(condensed_distance_matrix, method='complete')







# Create a dendrogram







plt.figure(figsize=(8, 6))







dendrogram(Z, labels=['a', 'b', 'c', 'd', 'e'], leaf_rotation=90)







plt.title("Dendrogram for Complete-Linkage Hierarchical Clustering")







plt.xlabel("Samples")







plt.ylabel("Distance")







plt.show()







X = linkage(condensed_distance_matrix, method='average')







plt.figure(figsize=(8, 6))







dendrogram(X, labels=['a', 'b', 'c', 'd', 'e'], leaf_rotation=90)







plt.title("Dendrogram for Average-Linkage Hierarchical Clustering")







plt.xlabel("Samples")







plt.ylabel("Distance")







plt.show()







Y = linkage(condensed_distance_matrix, method='single')







# Create a dendrogram







plt.figure(figsize=(8, 6))







dendrogram(Y, labels=['a', 'b', 'c', 'd', 'e'], leaf_rotation=90)







plt.title("Dendrogram for single-Linkage Hierarchical Clustering")







plt.xlabel("Samples")







plt.ylabel("Distance")







plt.show()







