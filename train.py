import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("bhp.csv")

# Drop missing values
df = df.dropna()

# Convert BHK (if necessary)
df['BHK'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else x)

# Select relevant columns
df = df[['total_sqft', 'bath', 'BHK', 'price']]

# Define features & target
X = df[['total_sqft', 'bath', 'BHK']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model R² Score: {r2_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "model.pkl")
print("Random Forest model trained & saved!")
