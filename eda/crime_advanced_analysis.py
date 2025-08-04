# crime_advanced_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np

# Settings
sns.set(style="whitegrid")
plt.rcParams["font.size"] = 11

# Load data
crime_details = pd.read_csv('/Users/sudearici/Documents/crime_prediction/data/update_crime.csv', sep=';')

selected_crimes = [
    "assault", "theft", "robbery",
    "production_and_commerce_of_drugs",
    "opposition_to_the_bankruptcy_and_enforcement_law"
]
crime_details = crime_details[crime_details["crime_type"].isin(selected_crimes)]

# AGE GROUP ANALYSIS
age_cols = ["12_14_age", "15_17_age", "18_24_age", "25_34_age", "35_44_age", "45_54_age", "55_64_age", "65+_age"]
age_dist = crime_details.groupby(["crime_type", "year"])[age_cols].sum().reset_index()
for crime in selected_crimes:
    temp = age_dist[age_dist["crime_type"] == crime].drop("crime_type", axis=1).set_index("year")
    temp.set_index(temp.index.astype(str), inplace=True)
    temp[age_cols].plot(kind="line", figsize=(12,6), marker="o")
    plt.title(f"Age Distribution Over Years ({crime})")
    plt.ylabel("Number of Crimes")
    plt.xlabel("Year")
    plt.legend(title="Age Group")
    plt.tight_layout()
    plt.show()

# EDUCATION ANALYSIS
e_cols = ["illiterate", "literate_but_not_graduated_from_a_school", "primary_school", "primary_education",
          "junior_high_school_and_vocational_school_at_high_school_level",
          "high_school_and_vocational_school_at_high_school_level", "higher_education"]
edu_dist = crime_details.groupby(["crime_type", "year"])[e_cols].sum().reset_index()
for crime in selected_crimes:
    temp = edu_dist[edu_dist["crime_type"] == crime].drop("crime_type", axis=1).set_index("year")
    temp.set_index(temp.index.astype(str), inplace=True)
    temp[e_cols].plot(kind="bar", stacked=True, figsize=(13,6))
    plt.title(f"Education Distribution Over Years ({crime})")
    plt.ylabel("Number of Crimes")
    plt.xlabel("Year")
    plt.legend(title="Education Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# MARITAL STATUS ANALYSIS
m_cols = ["never_married", "married", "widowed", "divorced"]
marital_dist = crime_details.groupby(["crime_type", "year"])[m_cols].sum().reset_index()
for crime in selected_crimes:
    temp = marital_dist[marital_dist["crime_type"] == crime].drop("crime_type", axis=1).set_index("year")
    temp.set_index(temp.index.astype(str), inplace=True)
    temp[m_cols].plot(kind="line", marker="o", figsize=(10,5))
    plt.title(f"Marital Status Distribution Over Years ({crime})")
    plt.ylabel("Number of Crimes")
    plt.xlabel("Year")
    plt.legend(title="Marital Status")
    plt.tight_layout()
    plt.show()

# CORRELATION ANALYSIS
numerical_cols = age_cols + e_cols + m_cols
corr = crime_details[numerical_cols].corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# NORMALIZATION AND CLUSTERING
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(crime_details[numerical_cols])
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(scaled_data)
crime_details["cluster"] = labels

# CLUSTER VISUALIZATION
cluster_avg = crime_details.groupby("cluster")[numerical_cols].mean().T
cluster_avg.plot(kind="bar", figsize=(15,6))
plt.title("Average Values per Cluster")
plt.ylabel("Normalized Average")
plt.xlabel("Variables")
plt.tight_layout()
plt.show()

# CITY-BASED GEOGRAPHIC ANALYSIS (if 'city' column exists)
if "city" in crime_details.columns:
    city_total = crime_details.groupby("city")["total_crime"].sum().reset_index()
    fig = px.choropleth(city_total, locations="city", locationmode="country names",
                        color="total_crime", title="Crime Intensity by Cities")
    fig.show()

# Time series plot for each crime type
for crime in selected_crimes:
    df_temp = crime_details[crime_details["crime_type"] == crime].groupby("year")["total_crime"].sum().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(df_temp["year"], df_temp["total_crime"], marker="o", linewidth=2)
    plt.title(f"Yearly Total Number of Crimes for {crime.replace('_', ' ').title()}", fontsize=13, weight='bold')
    plt.xlabel("Year")
    plt.ylabel("Total Number of Crimes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
