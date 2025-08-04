import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from preprocessing.data_processing import df, numeric_cols, categorical_cols  # df'yi alıyoruz
# 🎯 Histogram
df[numeric_cols].hist(bins=20, figsize=(24, 20))
plt.suptitle("Data Distribution")
plt.show()

# 🎯 Boxplot
plt.figure(figsize=(24, 16))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=90)
plt.title("Outlier Analysis and Data Distribution")
plt.show()

# 🎯 Correlation Matrix
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(24, 16))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 🎯 Crime Distribution by Gender - Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="gender", y="total_crime", hue="gender", palette="Set2")
plt.legend().remove()
plt.title("Crime Distribution by Gender")
plt.show()

# 🎯 Crime Distribution by Gender - Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="gender", y="total_crime", hue="gender", palette="muted", inner="quartile")
plt.legend().remove()
plt.title("Crime Distribution by Gender (Violin Plot)")
plt.show()

# 🎯 Time Series - By Cities
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="year", y="total_crime", hue="city", alpha=0.5)
plt.legend().remove()
sns.lineplot(data=df.groupby("year")["total_crime"].mean(), label="Average", linewidth=3, color="black")
plt.title("Annual Crime Trends by City")
plt.grid(True)
plt.show()

# 🎯 Geographical Crime Distribution
city_crime = df.groupby("city")["total_crime"].sum().reset_index()
fig = px.bar(city_crime.sort_values("total_crime", ascending=False),
             x="total_crime", y="city", orientation="h",
             color="total_crime", title="Total Crime Distribution by City")
fig.show()

# 🎯 Total Violent Crimes
df["violent_crimes"] = df["homicide"] + df["assault"] + df["sexual_crimes"] + df["robbery"]
df_violent = df.groupby("city")["violent_crimes"].sum().reset_index()

plt.figure(figsize=(16, 10))
sns.barplot(y="city", x="violent_crimes", data=df_violent.sort_values("violent_crimes", ascending=False), palette="Oranges_r")
plt.title("Total Violent Crimes by City")
plt.show()

# 🎯 Total Crime by Year
plt.figure(figsize=(12, 6))
sns.lineplot(data=df.groupby("year")["total_crime"].sum().reset_index(), x="year", y="total_crime", marker="o", color="red")
plt.title("Total Crime by Year")
plt.grid(True)
plt.show()

# 🎯 Total Distribution of Crime Types
numeric_crime_cols = df.select_dtypes(include='number').columns.drop(['TR_ID', 'year', 'total_crime', 'violent_crimes', 'city_encoded'], errors="ignore")
crime_totals = df[numeric_crime_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
sns.barplot(x=crime_totals.values, y=crime_totals.index, palette="Blues_r")
plt.title("Total Distribution of Crime Types")
plt.xlabel("Total Crime Count")
plt.show()

# 🎯 Top 10 Cities with the Most Crimes
top_cities = df.groupby("city")["total_crime"].sum().nlargest(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x="total_crime", y="city", data=top_cities, palette="Reds_r")
plt.title("Top 10 Cities with the Most Crimes")
plt.show()

# 🎯 Categorical Variable Frequencies
for col in categorical_cols:
    print(f"{col} - Categorical Distribution:\n", df[col].value_counts(), "\n")