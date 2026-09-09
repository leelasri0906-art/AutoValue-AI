from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'car data.csv'
MODEL_PATH = BASE_DIR / 'models' / 'car_price_model.pkl'
CURRENT_YEAR = 2026

if not DATA_PATH.is_file():
    raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')

df = pd.read_csv(DATA_PATH)
required = ['name','year','selling_price','km_driven','fuel','seller_type','transmission','owner']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

df = df[required].dropna().copy()
df['age'] = (CURRENT_YEAR - pd.to_numeric(df['year'], errors='coerce')).clip(lower=0)
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df = df.dropna()
df = df[df['selling_price'] > 0]

df = df.rename(columns={
    'name':'Car_Name', 'selling_price':'Selling_Price', 'km_driven':'Kms_Driven',
    'fuel':'Fuel_Type', 'seller_type':'Seller_Type', 'transmission':'Transmission', 'owner':'Owner'
})
X = df[['Car_Name','age','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
y = df['Selling_Price']

categorical = ['Car_Name','Fuel_Type','Seller_Type','Transmission','Owner']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=1, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print('\nModel Evaluation')
print('----------------')
print(f'MAE  : ₹{mae:,.0f}')
print(f'RMSE : ₹{rmse:,.0f}')
print(f'R²   : {r2:.4f}')
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f'\nSaved model to: {MODEL_PATH}')
