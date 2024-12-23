import numpy as np
import pandas as pd
import glob
import datetime as dt
import time as tm

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Definir random state e métrica
seed = 1
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

scorer = make_scorer(rmse, greater_is_better=False)
cv = TimeSeriesSplit(n_splits=5)

#definir MAPE
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

scorer_mape = make_scorer(mape, greater_is_better=False)

# Novo conjunto de algoritmos
algorithms = {
    'Decision Tree': GridSearchCV(
        Pipeline([('scaler', RobustScaler()), ('tree', DecisionTreeRegressor(random_state=seed))]),
        param_grid={
            'tree__max_depth': [10, 20, 30],
            'tree__criterion': ['squared_error', 'friedman_mse'],
        },
        scoring=scorer_mape,
        cv=cv,
    ),
    'KNN': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())]),
        param_grid={
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['uniform', 'distance'],
        },
        scoring=scorer_mape,
        cv=cv,
    ),    
    'MLP': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(random_state=seed, max_iter=500))]),
        param_grid={
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu', 'tanh', 'logistic'],
        },
        scoring=scorer_mape,
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
        scoring=scorer_mape,
        cv=cv,
    ),
    'AdaBoost': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('ada', AdaBoostRegressor(random_state=seed))]),
        param_grid={
            'ada__n_estimators': [50, 100, 200],
            'ada__learning_rate': [0.01, 0.1, 1],
        },
        scoring=scorer_mape,
        cv=cv,
    ), 
    'Gradient Boosting': GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingRegressor(random_state=seed))]),
        param_grid={
            'gb__n_estimators': [50, 100, 200],
            'gb__learning_rate': [0.01, 0.1, 0.2],
            'gb__max_depth': [3, 5, 7],
        },
        scoring=scorer_mape,
        cv=cv,
    )

}

# Use a wildcard pattern to match any CSV file in the yahoo_data subfolder
file_path_pattern = 'yahoo_data/yahoo_data*.csv'

# Find all files matching the pattern
file_list = glob.glob(file_path_pattern)

# Check if any files matched
if file_list:
    # Get the first matching file (if there are multiple)
    file_path = file_list[-1]
    
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the data
    print(data.head(),'\n\n',data.tail())

else:
    print("No files found matching the pattern.")

days = 5

data = data.drop(columns=['Open', 'High', 'Low', 'Close'])
data['Future'] = data['Adjusted'].shift(-days)
data = data.dropna()

print(data.head(),'\n\n',data.tail())

tickers = data['stock'].unique()
print(tickers)

# Dictionary to store train and test sets for each ticker
ticker_train_test_data = {}

# Df to store predictions
predictions = pd.DataFrame(columns=['date', 'stock', 'Adjusted', 'Prediction', 'Model'])

# Define the test period (last 30 rows)
test_rows = 30
results = []
for ticker in data['stock'].unique():
    ticker_data = data[data['stock'] == ticker]
    print(ticker)

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

    print(train_data.head())

    # Executar cross-validation e coletar as melhores métricas
    results_ticker = []

    for name, model in algorithms.items():
        model.fit(train_data[['Adjusted', 'Volume']], train_data['Future'])
        y_pred = model.predict(test_data[['Adjusted', 'Volume']])
        rmse_value = rmse(test_data['Future'], y_pred)
        mape_value = mape(test_data['Future'], y_pred)
        results_ticker.append((name, rmse_value, mape_value, ticker))
        
        #save predictions
        predictions = pd.concat([predictions, pd.DataFrame({'stock': ticker, 'date': test_data['date'], 'Adjusted': test_data['Adjusted'] , 'Prediction': y_pred, 'Model': name})])
        print(predictions.tail())
    
    #print(test_data.head())
    results_ticker = pd.DataFrame(results_ticker, columns=['Algorithm', 'RMSE', 'MAPE', 'Ticker'])
    print(results_ticker)

    results.append(results_ticker)


print(results)
print('\n\n',predictions)

# Concatenate all results into a single DataFrame
final_results = pd.concat(results, ignore_index=True)

# Save the results to a CSV file
final_results.to_csv(path_or_buf='models_results/model_results.csv', index=False)
predictions.to_csv(path_or_buf='models_results/predictions.csv', index=False)

print("Results saved to model_results.csv, and predictions.csv")

#####################################

### Analysis of algorithm selection
model_results = pd.read_csv('models_results/model_results.csv')
best_model_each_stock = model_results.loc[model_results.groupby('Ticker')['MAPE'].idxmin()]
best_model_each_stock.index = best_model_each_stock['Ticker']
best_model_each_stock = best_model_each_stock.drop(columns=['Ticker'])
print("\nbest_model_each_stock:", "\n", best_model_each_stock.head(10))

group_results = best_model_each_stock.groupby('Algorithm').size().sort_values(ascending=False)
print("\ngroup_results:\n",group_results)


### Analysis of predictions
data_predictions = pd.read_csv('models_results/predictions.csv')
data_predictions['date'] = pd.to_datetime(data_predictions['date'])

# subset only best model for each ticker
def filter_best_model(group):
    """
    Filters the group to only include rows where the model matches the best model for the ticker.
filtered_data = data.groupby('Ticker').apply(filter_best_model)
print("\n",filtered_data.head(50))
    group (DataFrame): The group of data for a specific ticker.

    Returns:
    DataFrame: The filtered group with only the best model.
    """
    return group.loc[group['Model'] == best_model_each_stock.loc[group.name, 'Algorithm']]

data_predictions_best_model = data_predictions.groupby('stock').apply(filter_best_model).reset_index(drop=True)
print("\ndata_predictions_best_model:\n",data_predictions_best_model.head(50))


### Clear incomplete data
last_dates = data_predictions_best_model.groupby('stock')['date'].max()
#print("\nlast_dates:\n", last_dates)
max_date = last_dates.max()

complete_stocks = last_dates[last_dates == max_date].index
data_predictions_best_model = data_predictions_best_model[data_predictions_best_model['stock'].isin(complete_stocks)]
#print("\ndata_predictions_best_model:\n",data_predictions_best_model.tail(50))

####Backtest
backtest = []
interval = list(range(9, 50)) + [0]
for i in interval:
    print("date less:", i)
    date_selection = max_date - pd.Timedelta(days=i)

    ### Stocks selection
    analysis_df = data_predictions_best_model.copy()

    #Margin definition
    analysis_df['Margin_prediction'] = (analysis_df['Prediction'] - analysis_df['Adjusted'])/analysis_df['Adjusted']

    #RMSE filter
    analysis_df = analysis_df.merge(best_model_each_stock[['RMSE']], left_on='stock', right_index=True, how='left')
    analysis_df = analysis_df.merge(best_model_each_stock[['MAPE']], left_on='stock', right_index=True, how='left')
    analysis_df['rmse_rate'] = analysis_df['RMSE']/analysis_df['Adjusted']
    filtered_analysis_df = analysis_df[analysis_df['date'] == max_date].sort_values(by='MAPE')
    filtered_analysis_df = analysis_df[(analysis_df['date'] == max_date) & (analysis_df['MAPE'] <= 1)].sort_values(by='MAPE')
    #print("\nFiltered analysis_df by max_date and sorted by rmse_rate:\n", filtered_analysis_df.tail())

    # Drop rows from analysis_df if stock is not in filtered_analysis_df
    analysis_df = analysis_df[analysis_df['stock'].isin(filtered_analysis_df['stock'])]
    #print("\nanalysis_df after dropping stocks not in filtered_analysis_df:\n", analysis_df.tail())

    # signals
    analysis_df['Up_down'] = analysis_df['Margin_prediction'].apply(lambda x: 1 if x > 0 else 0)
    analysis_df['Selected'] = 0
    top_5_margin_predictions = analysis_df[analysis_df['date'] == date_selection].nlargest(5, 'Margin_prediction').index
    analysis_df.loc[top_5_margin_predictions, 'Selected'] = 1
    #print("\nanalysis_df with signals :\n", analysis_df.tail())
    #print("\nselected stocks df :\n", analysis_df.loc[top_5_margin_predictions])
    portfolio = analysis_df.loc[top_5_margin_predictions, 'stock'].values
    print("\nPortfolio stocks:\n", portfolio)

    ###Results Analysis
    analysis_df['Results'] = analysis_df['Adjusted'].pct_change(5).shift(-5)
    portfolio_df = analysis_df.loc[top_5_margin_predictions].sort_values(by='Margin_prediction', ascending=False)
    #print("\nresults df :\n", analysis_df.tail(20))
    print("\nportfolio_df :\n", portfolio_df)

    if i == 0:
        portfolio_stocks = portfolio
        portfolio_df['Margin_prediction'] = portfolio_df['Margin_prediction']
        predictions_prices = portfolio_df.set_index('stock')[['Adjusted', 'Prediction', 'Margin_prediction']]

    else:
        final_result = portfolio_df['Results'].sum()
        print("\nFinal result:\n", final_result)    
        backtest.append(final_result)

print("\nBacktest results:\n", backtest)

positive_results = [result for result in backtest if result > 0]
negative_results = [result for result in backtest if result < 0]

print("\nNumber of positive results:", len(positive_results))
print("Number of negative results:", len(negative_results))

positive_rate = len(positive_results) / (len(positive_results) + len(negative_results))
print("Positive rate:", positive_rate)

print("\nSum of backtest results:\n", sum(backtest))
print("\nPortfolio: ", portfolio_stocks, "\n\n",  predictions_prices)

### Df deploy
df_stocks = pd.DataFrame(columns=['date', 'stock', 'Prediction', 'Model', 'Margin_prediction'])
for ticker in portfolio_stocks:
    df_stocks = pd.concat([df_stocks, pd.DataFrame({
        'date': analysis_df[analysis_df['stock'] == ticker].tail(5)['date'].tolist(),
        'stock': analysis_df[analysis_df['stock'] == ticker].tail(5)['stock'].tolist(),
        'Prediction': analysis_df[analysis_df['stock'] == ticker].tail(5)['Prediction'].tolist(),
        'Model': analysis_df[analysis_df['stock'] == ticker].tail(5)['Model'].tolist(),
        'Margin_prediction': analysis_df[analysis_df['stock'] == ticker].tail(5)['Margin_prediction'].tolist()
    })])

#Adjust date and Margin_prediction
df_stocks['date'] = df_stocks.groupby('stock').cumcount() + 1
df_stocks['Margin_prediction'] = df_stocks.groupby('stock')['Margin_prediction'].transform('last')
print(df_stocks)

## Import all yahoo data for sortino and sharpe index
# Use a wildcard pattern to match any CSV file in the yahoo_data subfolder
file_path_pattern = 'yahoo_data/yahoo_data*.csv'

# Find all files matching the pattern
file_list = glob.glob(file_path_pattern)

# Check if any files matched
if file_list:
    # Get the first matching file (if there are multiple)
    file_path = file_list[-1]
    
    # Read the CSV file
    all_data = pd.read_csv(file_path)
    
    # Display the first few rows of the data
    print(all_data.head(),'\n\n',all_data.tail())

else:
    print("No files found matching the pattern.")

all_data = pd.read_csv(file_path, usecols=['date', 'stock', 'Adjusted'])
# Pivot the data to have stocks as columns
all_data_pivot = all_data.pivot(index='date', columns='stock', values='Adjusted')

# Calculate daily returns
all_data_pivot = all_data_pivot.pct_change().dropna()

# Calculate mean and standard deviation of returns for each stock
mean_returns = all_data_pivot.mean()
std_returns = all_data_pivot.std()

# Assuming a risk-free rate of 5% per year
risk_free_rate = 0.05 / 252  # Convert annual risk-free rate to daily

# Calculate Sharpe ratio for each stock
sharpe_ratios = (mean_returns - risk_free_rate) / std_returns
# Calculate portfolio returns
portfolio_returns = all_data_pivot[portfolio_stocks].mean(axis=1)

# Calculate portfolio Sharpe ratio
portfolio_sharpe_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std()

# Calculate downside deviation for Sortino ratio
downside_returns = all_data_pivot[all_data_pivot < 0].std()

# Calculate Sortino ratio for each stock
sortino_ratios = (mean_returns - risk_free_rate) / downside_returns
# Calculate portfolio Sortino ratio
portfolio_downside_returns = portfolio_returns[portfolio_returns < 0].std()
portfolio_sortino_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_downside_returns


# Add Sharpe and Sortino ratios to df_stocks
df_stocks['Sharpe'] = df_stocks['stock'].map(sharpe_ratios)
df_stocks['Sortino'] = df_stocks['stock'].map(sortino_ratios)

print(df_stocks)



###Margem, Sharpe e Sortino da Carteira

# Calculate the portfolio margin based
portfolio_margin = predictions_prices['Margin_prediction'].mean()

print("\nPortfolio Margin:\n", portfolio_margin)
print("\nPortfolio Sharpe Ratio:\n", portfolio_sharpe_ratio)
print("\nPortfolio Sortino Ratio:\n", portfolio_sortino_ratio)

portfolio_numbers = pd.DataFrame({
    'Metric': ['Margin', 'Sharpe Ratio', 'Sortino Ratio'],
    'Value': [portfolio_margin, portfolio_sharpe_ratio, portfolio_sortino_ratio]
})

print("\nPortfolio Numbers:\n", portfolio_numbers)

# Save the results to a CSV file
df_stocks.to_csv(path_or_buf='models_results/portfolio.csv', index=False)
portfolio_numbers.to_csv(path_or_buf='models_results/portfolio_numbers.csv', index=False)

# Save backtest results to a CSV file
backtest_results_df = pd.DataFrame({
    'Backtest Results': backtest,
    'Positive Results': positive_results + [None] * (len(backtest) - len(positive_results)),
    'Negative Results': negative_results + [None] * (len(backtest) - len(negative_results)),
    'Number of Positive Results': [len(positive_results)] * len(backtest),
    'Number of Negative Results': [len(negative_results)] * len(backtest),
    'Positive Rate': [positive_rate] * len(backtest)
})

backtest_results_df.to_csv('models_results/backtest_results.csv', index=False)
