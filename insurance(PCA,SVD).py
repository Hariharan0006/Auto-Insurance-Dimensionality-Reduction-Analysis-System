'''
# Dimension Reduction - PCA
# PCA helps in reducing dimensionality by transforming correlated features
# into a smaller set of uncorrelated principal components.

# CRISP-ML(Q):

Business & Data Understanding:
    Business Problem:
        Insurance dataset has many numerical variables influencing customer behavior.
        High dimensionality increases computation time and complexity.

    Business Objective:
        Reduce dimensionality while retaining maximum information.

    Business Constraints:
        Do not lose important customer behavior information.

Success Criteria:
    Business:
        Improve customer segmentation and targeting.
    ML:
        Achieve at least 70% variance retention.
    Economic:
        Improve cross-sell revenue by 8–10%.

# Data Source:
# Autoinsurance.csv

# Data Dictionary (Sample):
# Customer          : Customer ID
# State             : Customer state
# Coverage          : Insurance coverage type
# Education         : Education level
# Income            : Annual income
# Monthly Premium   : Monthly insurance premium
# Total Claim Cost  : Total claim amount
# Vehicle Size      : Vehicle size
'''
# !pip install sweetviz
# !pip install kneed

import numpy as np
import pandas as pd
import sweetviz
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from kneed import KneeLocator
import joblib

df = pd.read_csv(r"Dataset/AutoInsurance.csv")
df.head()

df.info()
df.describe()

# Dropping identifier columns
df1 = df.drop(['Customer'], axis=1)

# Check missing values
df1.isnull().sum()

report = sweetviz.analyze(df1)
report.show_html("AutoInsurance_EDA.html")

numeric_features = df1.select_dtypes(exclude=['object']).columns
numeric_features

pca = PCA(n_components=10)

num_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    pca
)

processed = num_pipeline.fit(df1[numeric_features])

pca_res = pd.DataFrame(processed.transform(df1[numeric_features]))
pca_res.head()

joblib.dump(processed, 'AutoInsurance_PCA_Model')

print(processed['pca'].explained_variance_ratio_)

cumulative_variance = np.cumsum(processed['pca'].explained_variance_ratio_)
print(cumulative_variance)

print(processed['pca'].explained_variance_ratio_)

cumulative_variance = np.cumsum(processed['pca'].explained_variance_ratio_)
print(cumulative_variance)

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print("Selected components:", n_components)

final_pca = pd.concat(
    [df[['Customer']], pca_res.iloc[:, :n_components]],
    axis=1
)

final_pca.head()


'''
# Dimension Reduction - SVD
# SVD is useful for large sparse datasets and works without covariance matrix.

Business Objective:
    Reduce data size while preserving structure.

ML Objective:
    Feature extraction using TruncatedSVD.
'''
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=10)

svd_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    svd
)

svd_model = svd_pipeline.fit(df1[numeric_features])

svd_res = pd.DataFrame(svd_model.transform(df1[numeric_features]))
svd_res.head()

joblib.dump(svd_model, 'AutoInsurance_SVD_Model')

print(svd.explained_variance_ratio_)
svd_cum_var = np.cumsum(svd.explained_variance_ratio_)
print(svd_cum_var)

plt.plot(svd_cum_var, color='blue')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

final_svd = pd.concat([df[['Customer']], svd_res.iloc[:, :3]], axis=1)
final_svd.head()
