import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from catboost import CatBoostRegressor

# Load dataset
data = pd.read_csv("antenna_dataset.csv")

X = data[['R1','R2','R3','R4','R5','R6','R7','R8','d','Wf']]
y = data[['F1','F2','F3','BW1','BW2','BW3']]

# Train test split R²

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("\nDataset Loaded Successfully")
print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))


models = {
"Linear Regression": LinearRegression(),
"CatBoost":          CatBoostRegressor(verbose=0, random_seed=42),
"Gradient Boosting": GradientBoostingRegressor(random_state=42),
"Extra Trees":       ExtraTreesRegressor(n_estimators=200, random_state=42),
"Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
"Decision Tree":     DecisionTreeRegressor(random_state=42)
}

results = []
target_cols = ['F1','F2','F3','BW1','BW2','BW3']
trained_models = {}

print("\nTraining Models...\n")
print(f"{'Model':<22} {'Avg R²':>8}  {'F1':>7} {'F2':>7} {'F3':>7} {'BW1':>7} {'BW2':>7} {'BW3':>7}")
print("-" * 80)

for name,model in models.items():

    wrapped_model = MultiOutputRegressor(model)
    wrapped_model.fit(X_train,y_train)

    pred = wrapped_model.predict(X_test)

    per_output = r2_score(y_test, pred, multioutput='raw_values')
    avg_r2 = per_output.mean()

    print(f"{name:<22} {avg_r2:>8.4f}  {per_output[0]:>7.4f} {per_output[1]:>7.4f} {per_output[2]:>7.4f} {per_output[3]:>7.4f} {per_output[4]:>7.4f} {per_output[5]:>7.4f}")

    results.append((name,avg_r2))
    trained_models[name] = wrapped_model


# Sort Results

print("\n" + "="*80)
print("Model Ranking (by Avg R²):")
print("="*80)

results = sorted(results,key=lambda x:x[1],reverse=True)

for i,(model,r2) in enumerate(results, 1):
    print(f"  {i}. {model:<22} R2 = {r2:.4f}")



# Prediction

print("\n" + "="*50)
best_model_name = results[0][0]
print(f"Prediction using Best Model: {best_model_name}")
print("="*50)
print("Enter the antenna parameters below:\n")

best_model = trained_models[best_model_name]

features = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'd', 'Wf']
input_values = {}

for feat in features:
    while True:
        try:
            val = float(input(f"  {feat}: "))
            input_values[feat] = val
            break
        except ValueError:
            print(f"  Invalid input. Please enter a numeric value for {feat}.")

sample = pd.DataFrame([input_values])

prediction = best_model.predict(sample)

print("\nInput Parameters:")
for feat, val in input_values.items():
    print(f"  {feat} = {val}")

print("\nPredicted Output:")
output_labels = ['F1 (GHz)', 'F2 (GHz)', 'F3 (GHz)', 'BW1 (GHz)', 'BW2 (GHz)', 'BW3 (GHz)']
for label, val in zip(output_labels, prediction[0]):
    print(f"  {label} = {val:.6f}")