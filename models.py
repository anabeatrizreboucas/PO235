import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import glob

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, mean_squared_error
from math import sqrt

# Definir random state e métrica
seed = 1
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

scorer = make_scorer(rmse, greater_is_better=False)
cv = TimeSeriesSplit(n_splits=5)

# Novo conjunto de algoritmos
algorithms = {
    'Decision Tree': GridSearchCV(
        Pipeline([('scaler', RobustScaler()), ('tree', DecisionTreeRegressor(random_state=seed))]),
        param_grid={
            'tree__max_depth': [10, 20, 30],
            'tree__criterion': ['squared_error', 'friedman_mse'],
        },
        scoring=scorer,
        cv=cv,
    ),
    'KNN': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())]),
        param_grid={
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['uniform', 'distance'],
        },
        scoring=scorer,
        cv=cv,
    ),
    'MLP': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(random_state=seed, max_iter=500))]),
        param_grid={
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu', 'tanh', 'logistic'],
        },
        scoring=scorer,
        cv=cv,
    ),
    'Random Forest': GridSearchCV(
        Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=seed))]),
        param_grid={
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [10, 20],
        },
        scoring=scorer,
        cv=cv,
    ),
    'AdaBoost': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('ada', AdaBoostRegressor(random_state=seed))]),
        param_grid={
            'ada__n_estimators': [50, 100, 200],
            'ada__learning_rate': [0.01, 0.1, 1],
        },
        scoring=scorer,
        cv=cv,
    ),
    'Gradient Boosting': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingRegressor(random_state=seed))]),
        param_grid={
            'gb__n_estimators': [50, 100, 200],
            'gb__learning_rate': [0.01, 0.1, 0.2],
            'gb__max_depth': [3, 5, 7],
        },
        scoring=scorer,
        cv=cv,
    ),
}

# Use a wildcard pattern to match any CSV file in the yahoo_data subfolder
file_path_pattern = 'yahoo_data/yahoo_data*.csv'

# Find all files matching the pattern
file_list = glob.glob(file_path_pattern)

# Check if any files matched
if file_list:
    # Get the first matching file (if there are multiple)
    file_path = file_list[0]
    
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the data
    print(data.head())

else:
    print("No files found matching the pattern.")

## create columns of results
data['result'] = data.groupby('stock')['Adjusted'].pct_change()
data['result'] = data.groupby('stock')['result'].shift(-1) 

data['result_categorical'] = np.where(data['result'] > 0, 1, 0)

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna(subset=['log_return'])


print(data.head())

tickers = data['stock'].unique()
# Append ".SA" to each ticker
#tickers_sa = [ticker + '.SA' for ticker in tickers]

#print(tickers)
#print(tickers_sa)

## Dictionary for tickers df
ticker_data = {}

# Dictionary to store train and test sets for each ticker
ticker_train_test_data = {}

# Define the test period (last 30 rows)
test_rows = 30
results = []
for ticker in data['stock'].unique():
    ticker_data = data[data['stock'] == ticker]

    # Check if there are enough rows for a 30-row test set
    if len(ticker_data) > test_rows:
        # Split into train and test
        train_data = ticker_data.iloc[:-test_rows]
        test_data = ticker_data.iloc[-test_rows:]
    else:
        # If not enough data, use all data for training and leave test empty
        train_data = ticker_data
        test_data = pd.DataFrame()  # Empty DataFrame for test set
        print("There is NO data enough for train and test")
    
    # Store train and test sets in the dictionary
    ticker_train_test_data[ticker] = {
        'train': train_data,
        'test': test_data
    }

    # Executar cross-validation e coletar as melhores métricas
    results_ticker = []

    for name, model in algorithms.items():
        model.fit(train_data[['log_return']], train_data['log_return'])
        y_pred = model.predict(test_data[['log_return']])
        rmse_value = rmse(test_data['log_return'], y_pred)
        results_ticker.append((name, rmse_value, ticker))

    results_ticker = pd.DataFrame(results, columns=['Algorithm', 'RMSE', 'Ticker'])
    print(results_ticker)
    results.append(results_ticker)
print(results)
'''
# Treinar o melhor modelo
best_model = algorithms[results.loc[results['RMSE'].idxmin(), 'Algorithm']]
best_model.fit(train_data[['log_return']], train_data['result'])

# Prever o resultado
y_pred = best_model.predict(test_data[['log_return']])
test_data['predicted_result'] = y_pred

# Plotar o resultado
plt.figure(figsize=(10, 5))
plt.plot(test_data['Date'], test_data['result'], label='Real')
plt.plot(test_data['Date'], test_data['predicted_result'], label='Predicted')
plt.title(f'{ticker} - Result Prediction')
plt.legend()

plt.show()

    
# Salvar o modelo
model_name = f'{ticker}_model.pkl'
joblib.dump(best_model, model_name)

print(f'Model saved as {model_name}')
'''