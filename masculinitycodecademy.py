import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

survey = pd.read_csv("masculinity.csv")
#print(survey.columns)
#print(len(survey))

cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009","q0007_0010", "q0007_0011"]
for i in cols_to_map:
    survey[i] = survey[i].map({"Never, and not open to it": 0, "Never, but open to it": 1,"Rarely":2,"Sometimes":3,"Often":4})

print(survey['q0007_0001'].value_counts())

#plt.scatter(survey["q0007_0001"],survey["q0007_0002"],alpha=0.1)
#plt.show()

rows_to_cluster=survey.dropna(subset=["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005", "q0007_0008", "q0007_0009"])

model=KMeans(n_clusters=2)
model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005", "q0007_0008", "q0007_0009"]])
#print(model.cluster_centers_)
print(model.labels_)
cluster_zero_indices=[]
cluster_one_indices=[]
for i in range(len(model.labels_)):
    if model.labels_[i]==0:
        cluster_zero_indices.append(i)
    elif model.labels_[i]==1:
        cluster_one_indices.append(i)

#print(cluster_zero_indices)
cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]
print(cluster_zero_df["educ4"].value_counts())
print(cluster_one_df["educ4"].value_counts())
