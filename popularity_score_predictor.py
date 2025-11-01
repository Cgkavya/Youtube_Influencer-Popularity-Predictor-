# %%
# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# %%
# Load the data
data = pd.read_excel(
    r"C:\Users\Kavya\OneDrive\Documents\DSA_internship\influencer_data.xlsx"
)

# %%
data.head(10)

# %%
data.shape

# %%
data.columns.tolist()

# %%
data.info()

# %%
data.describe()

# %%
# Checking for null values
data.isnull().sum()

# %%
# Removing null values, duplicates, unecessary features
data = data.drop_duplicates()
data = data.dropna()
data = data.drop(["channelId", "description", "publishedAt"], axis=1)

# %%
data.head(5)

# %%
data.shape

# %%
data.isnull().sum()

# %%
# Correlation heat map
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()

# %%

num_data = data.select_dtypes(include="number")
cat_data = data.select_dtypes(include="object")

# Identify categorical and numerical columns
num_cols = num_data.columns.tolist()
cat_cols = cat_data.columns.tolist()

print("numerical columns: ", num_cols)
print("categorical columns: ", cat_cols)

# %%
# Perform an outlier detection analysis on numerical variables (e.g., using the IQR method).
num_data.boxplot()
plt.xticks(rotation=45)
plt.show()

# %%
# Remove outliers from these features if they are not representative of typical house prices.


def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    data[column_name] = data[column_name].clip(upper=upper_bound)
    data[column_name] = data[column_name].clip(lower=lower_bound)
    return data[column_name]


for col in num_cols:
    num_data[col] = remove_outliers(num_data, col)
    num_data.boxplot()
    plt.xticks(rotation=45)
    plt.show()

# %%
# Distribution of Engagement Rate across different influencer types
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="query", y="EngagementRate", hue="query", data=data, palette="Set3", legend=False
)

plt.title("Engagement Rate by Influencer Type", fontsize=14)
plt.xlabel("Influencer Type", fontsize=12)
plt.ylabel("Engagement Rate", fontsize=12)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# %%
# Distribution of influencer counts across different categories
plt.figure(figsize=(10, 6))
sns.countplot(x="query", hue="query", data=data, palette="Set2", legend=False)

plt.title("Count of Influencers by Category", fontsize=14)
plt.xlabel("Influencer Type", fontsize=12)
plt.ylabel("Count", fontsize=12)

# 🔄 Rotate x-axis labels for clarity
plt.xticks(rotation=45, ha="right")  # or rotation=60 for steeper angle

plt.tight_layout()
plt.show()

# %%
# Engagement Rate distribution across influencer size categories.
subgroup = pd.cut(
    data["subscriberCount"],
    bins=[0, 10000, 100000, 1000000, 10000000],
    labels=["Micro", "Mid", "Macro", "Mega"],
)

plt.figure(figsize=(10, 6))
sns.violinplot(
    x=subgroup,  # use the temporary variable
    y=data["EngagementRate"],
    palette="Set3",
    hue=subgroup,
    legend=False,
)

plt.title("Engagement Rate by Influencer Size", fontsize=14)
plt.xlabel("Influencer Size Category", fontsize=12)
plt.ylabel("Engagement Rate", fontsize=12)
plt.show()


# %%
# Distribution of influencers’ video posting frequency
plt.figure(figsize=(8, 5))
sns.histplot(data["PostFrequency"], bins=20, kde=True, color="teal")

plt.title("Distribution of Video Posting Frequency", fontsize=14)
plt.xlabel("Videos per Month", fontsize=12)
plt.ylabel("Number of Influencers", fontsize=12)
plt.tight_layout()
plt.show()


# %%
# Relationship between subscriber count and total views across different influencer categories.
plt.figure(figsize=(8, 6))
sns.scatterplot(x="subscriberCount", y="viewCount", hue="query", data=data, alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.title("Subscriber Count vs Total Views", fontsize=14)
plt.xlabel("Subscriber Count (log scale)")
plt.ylabel("Total Views (log scale)")
plt.tight_layout()
plt.show()


# %%
# Encoding
cat_cols = ["query"]
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# %%
# Engagement per 1000 subscribers
data["engagement_per_sub"] = (data["avgLikes"] + data["avgComments"]) / (
    data["subscriberCount"] / 1000
)

# Views per subscriber (audience reach efficiency)
data["views_per_sub"] = data["viewCount"] / data["subscriberCount"]

# Base popularity = combination of organic reach & engagement
data["popularity_score"] = (
    0.6 * data["views_per_sub"].rank(pct=True)
    + 0.4 * data["engagement_per_sub"].rank(pct=True)
) * 9 + 1


print("✅ Popularity scores (1–10) added successfully!")
print(data["popularity_score"].describe())

# %%
features = [
    "subscriberCount",
    "viewCount",
    "videoCount",
    "avgLikes",
    "avgComments",
    "EngagementRate",
    "PostFrequency",
    "accountAgeDays",
]
X = data[features]
y = data["popularity_score"]


# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# %%
# 4️⃣ Define models
# --------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "R²": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
        }
    )

results_df = pd.DataFrame(results)
print("\n📈 Model Comparison Results:")
print(results_df.sort_values("R²", ascending=False))

# %%
# --- Step 1: Define base model ---
gbr = GradientBoostingRegressor(random_state=42)

# --- Step 2: Define parameter grid ---
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "subsample": [0.8, 1.0],
}

# --- Step 3: Define CV setup ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Step 4: Run GridSearchCV (with built-in CV) ---
grid_search = GridSearchCV(
    estimator=gbr, param_grid=param_grid, scoring="r2", cv=cv, n_jobs=-1, verbose=2
)

grid_search.fit(X_train, y_train)

# --- Step 5: Best parameters & CV score ---
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation R²: {grid_search.best_score_:.4f}")

# --- Step 6: Evaluate on test data ---
best_gbr = grid_search.best_estimator_
y_pred = best_gbr.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Final Model Performance on Test Set")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# %%
# 6️⃣ Feature Importance
# ==============================
importances = best_gbr.feature_importances_
feat_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
    by="Importance", ascending=False
)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance for Popularity Prediction")
plt.tight_layout()
plt.show()

# %%
# Predict and Rank Influencers
# ==============================
data["PredictedPopularity"] = best_gbr.predict(X)
top_popular = data.sort_values("PredictedPopularity", ascending=False).head(10)

print("\n🏆 Top 10 Influencers by Predicted Popularity:\n")
print(top_popular[["title", "subscriberCount", "viewCount", "PredictedPopularity"]])

plt.figure(figsize=(8, 5))
sns.barplot(x="PredictedPopularity", y="title", data=top_popular)
plt.title("Top 10 Influencers by Predicted Popularity Score")
plt.xlabel("Predicted Popularity")
plt.ylabel("Influencer")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.histplot(data["popularity_score"], bins=20, kde=True, color="green", alpha=0.6)
plt.title("Predicted Popularity Score Distribution (1–10)", fontsize=14)
plt.xlabel("Popularity Score")
plt.ylabel("Number of Influencers")
plt.grid(alpha=0.3)
plt.show()


# %%
import pickle

# Save the final tuned model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_gbr, f)

# Save the scaler (if you used one earlier)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open("model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, scaler, and features saved successfully!")


# %%
