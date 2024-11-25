import numpy as np







import matplotlib.pyplot as plt







from sklearn.cluster import DBSCAN







from collections import defaultdict















# Define the data points (using arbitrary coordinates for visualization)







data = np.array([







 [3, 7], [4, 6], [5, 5], [6, 4], [7, 3],







 [6, 2], [7, 2], [8, 4], [3, 3], [2, 6],







 [3, 5], [2, 4]







])















# Run DBSCAN







eps = 1.9 # Adjust this value to get desired clustering







min_samples = 4 # Adjust this value to get desired core point identification







dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)















# Get labels and core sample indices







labels = dbscan.labels_







core_sample_indices = dbscan.core_sample_indices_















# Identify core, border, and noise points







core_points = set(core_sample_indices)







border_points = set()







noise_points = set()















for i, label in enumerate(labels):







 if i not in core_points:







  if label != -1:







   border_points.add(i)







  else:







   noise_points.add(i)















# Find connections between points







connections = defaultdict(list)







for i in range(len(data)):







 for j in range(i + 1, len(data)):







  if np.linalg.norm(data[i] - data[j]) <= eps:







   connections[i].append(j)







   connections[j].append(i)















# Plotting







plt.figure(figsize=(12, 8))







colors = ['red' if i in core_points else 'blue' if i in border_points else 'gray' for i in range(len(data))]







plt.scatter(data[:, 0], data[:, 1], c=colors, s=100)















for i, (x, y) in enumerate(data):







 plt.annotate(f'P{i + 1}', (x, y), xytext=(5, 5), textcoords='offset points')















for i, connected in connections.items():







 for j in connected:







  plt.plot([data[i][0], data[j][0]], [data[i][1], data[j][1]], 'k-', alpha=0.3)















plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")







plt.xlabel("X-axis")







plt.ylabel("Y-axis")







plt.grid(True)







plt.show()















# Print analysis







print("Point Classifications:")







for i in range(len(data)):







 if i in core_points:







  print(f"P{i + 1}: Core")







 elif i in border_points:







  core_neighbors = [f"P{j + 1}" for j in connections[i] if j in core_points]







  print(f"P{i + 1}: Border (Part of Core {', '.join(core_neighbors)})")







 else:







  print(f"P{i + 1}: Noise (Not a part of any Core)")















print("\nConnections:")







for i, connected in connections.items():







 print(f"P{i + 1}: {', '.join([f'P{j + 1}' for j in connected])}")































































































































