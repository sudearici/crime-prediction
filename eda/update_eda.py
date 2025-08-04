import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📊 Visualization settings
sns.set(style="whitegrid")
plt.rcParams["font.size"] = 11

# 📌 Load the data
crime_details = pd.read_csv('/Users/sudearici/Documents/crime_prediction/data/update_crime.csv', sep=';')

# 🔎 Filter for the top 5 most committed crime types
selected_crimes = [
    "assault",
    "theft",
    "robbery",
    "production_and_commerce_of_drugs",
    "opposition_to_the_bankruptcy_and_enforcement_law"
]
crime_details = crime_details[crime_details["crime_type"].isin(selected_crimes)]

# 🔢 Convert numeric columns (convert text to numbers, fill missing with 0)
numeric_cols = crime_details.columns.drop(["crime_type", "gender", "year"])
crime_details[numeric_cols] = crime_details[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

### 1️⃣ AGE GROUP ANALYSIS ###
age_cols = ["12_14_age", "15_17_age", "18_24_age", "25_34_age", "35_44_age", "45_54_age", "55_64_age", "65+_age"]

# a) Age distribution by crime type
age_crime = crime_details.groupby("crime_type")[age_cols].sum().T
age_crime.plot(kind="bar", figsize=(13, 6), colormap="Set2")
plt.title("Age Distribution by Crime Type")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# b) Age groups over the years
age_year = crime_details.groupby("year")[age_cols].sum()
age_year.plot(kind="line", figsize=(12, 6), marker='o')
plt.title("Number of Crimes by Age Group Over the Years")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# c) Age distribution by gender
age_gender = crime_details.groupby("gender")[age_cols].sum().T
age_gender.plot(kind="bar", figsize=(12, 6), colormap="coolwarm")
plt.title("Age Distribution by Gender")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 2️⃣ MARITAL STATUS ANALYSIS ###
marital_cols = ["never_married", "married", "widowed", "divorced"]

# a) Marital status by crime type
marital_crime = crime_details.groupby("crime_type")[marital_cols].sum().T
marital_crime.plot(kind="bar", figsize=(10, 6), colormap="Pastel1")
plt.title("Marital Status Distribution by Crime Type")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# b) Marital status over the years
marital_year = crime_details.groupby("year")[marital_cols].sum()
marital_year.plot(kind="line", figsize=(12, 6), marker='o')
plt.title("Number of Crimes by Marital Status Over the Years")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# c) Marital status by gender
marital_gender = crime_details.groupby("gender")[marital_cols].sum().T
marital_gender.plot(kind="bar", figsize=(10, 6), colormap="cool")
plt.title("Marital Status Distribution by Gender")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

### 3️⃣ EDUCATIONAL STATUS ANALYSIS ###
edu_cols = [
    "illiterate",
    "literate_but_not_graduated_from_a_school",
    "primary_school",
    "primary_education",
    "junior_high_school_and_vocational_school_at_high_school_level",
    "high_school_and_vocational_school_at_high_school_level",
    "higher_education"
]

# a) Educational status by crime type
edu_distribution = crime_details.groupby("crime_type")[edu_cols].sum().T.reset_index()
edu_distribution = pd.melt(edu_distribution, id_vars="index", var_name="Crime Type", value_name="Number of Crimes")
edu_distribution.rename(columns={"index": "Education Level"}, inplace=True)

plt.figure(figsize=(14, 6))
sns.barplot(data=edu_distribution, y="Education Level", x="Number of Crimes", hue="Crime Type", palette="Set1")
plt.title("Education Level Distribution by Crime Type", fontsize=14, weight='bold')
plt.xlabel("Total Number of Crimes")
plt.ylabel("Education Level")
plt.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# b) Educational status over the years
edu_year = crime_details.groupby("year")[edu_cols].sum()
edu_year.plot(kind="line", figsize=(13, 6), marker='o')
plt.title("Number of Crimes by Education Level Over the Years")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# c) Educational status by gender
edu_gender = crime_details.groupby("gender")[edu_cols].sum().T
edu_gender.plot(kind="bar", figsize=(13, 6), colormap="coolwarm")
plt.title("Education Level Distribution by Gender")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
