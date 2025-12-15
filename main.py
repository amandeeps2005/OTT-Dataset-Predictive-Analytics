import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.frequent_patterns import apriori, association_rules

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("--- LOADING AND PREPROCESSING DATA ---")

df = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Desktop\\Python for class\\third year\\project\\OTT_Data_500_Rows.csv')

df.columns = [
    'Timestamp', 'Age', 'Gender', 'Year', 'Subscription_Type', 
    'Num_Apps', 'Weekday_Hours', 'Weekend_Hours', 'Device', 
    'Monthly_Spend', 'Genres', 'Binge_Watcher', 'Sleep_Effect', 'Email'
]

df = df.drop(columns=['Email'], errors='ignore')

numeric_cols = ['Age', 'Num_Apps', 'Weekday_Hours', 'Weekend_Hours', 'Monthly_Spend', 'Sleep_Effect']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['Total_Weekly_Hours'] = (df['Weekday_Hours'] * 5) + (df['Weekend_Hours'] * 2)

print(f"Original Row Count: {len(df)}")

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for col in ['Monthly_Spend', 'Total_Weekly_Hours', 'Num_Apps', 'Age']:
    df = remove_outliers(df, col)

print(f"Row Count after Cleaning: {len(df)}")
print("-" * 50)

print("--- GENERATING EDA PLOTS ---")

plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], kde=True, bins=5, color='skyblue')
plt.title('Distribution of Student Age (Cleaned)')
plt.show()

df_corr = df[numeric_cols]
plt.figure(figsize=(10, 8))

ax = sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Correlation Matrix')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

print("-" * 50)

print("--- UNIT II: REGRESSION ANALYSIS ---")

X_reg = df[['Num_Apps', 'Total_Weekly_Hours', 'Age']]
y_reg = df['Monthly_Spend']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_r, y_pred_r))
print("Mean Squared Error (MSE):", mean_squared_error(y_test_r, y_pred_r))
print("R2 Score:", r2_score(y_test_r, y_pred_r))

plt.figure(figsize=(6, 4))
plt.scatter(y_test_r, y_pred_r, color='green')
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'k--', lw=2)
plt.xlabel('Actual Spend')
plt.ylabel('Predicted Spend')
plt.title('Regression: Actual vs Predicted')
plt.show()

print("-" * 50)

print("--- UNIT III: CLASSIFICATION ANALYSIS ---")

le = LabelEncoder()
df['Gender_Code'] = le.fit_transform(df['Gender'])
y_class = df['Binge_Watcher'].apply(lambda x: 1 if x == 'Yes' else 0)
X_class = df[['Weekend_Hours', 'Sleep_Effect', 'Gender_Code']]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

print("Classification Report:")
print(classification_report(y_test_c, y_pred_c))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print("-" * 50)

print("--- UNIT IV: K-MEANS CLUSTERING ---")

X_cluster = df[['Weekday_Hours', 'Weekend_Hours']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Weekday_Hours', y='Weekend_Hours', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Student Clusters based on Viewing Habits')
plt.show()

print("-" * 50)

print("--- UNIT IV: ASSOCIATION RULE MINING ---")

genres_split = df['Genres'].str.get_dummies(sep=', ')

frequent_itemsets = apriori(genres_split, min_support=0.1, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence'], ascending=False)
    
    print("Top 5 Association Rules found:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
else:
    print("No frequent patterns found (Try lowering min_support if needed).")

print("\n--- PROJECT COMPLETE ---")
