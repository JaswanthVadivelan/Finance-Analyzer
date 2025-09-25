"""
Expense forecasting module using machine learning and statistical models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class ExpenseForecaster:
    """Forecasts future expenses using various ML and statistical models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.forecast_results = {}
    
    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for time series forecasting.
        
        Args:
            df: Transaction dataframe
            
        Returns:
            pd.DataFrame: Time series data aggregated by day/month
        """
        if 'date' not in df.columns or 'amount' not in df.columns:
            return pd.DataFrame()
        
        # Filter expenses only
        expenses_df = df[df['amount'] < 0].copy()
        expenses_df['abs_amount'] = expenses_df['amount'].abs()
        
        # Aggregate by date
        daily_expenses = expenses_df.groupby('date')['abs_amount'].sum().reset_index()
        daily_expenses = daily_expenses.sort_values('date')
        
        # Create complete date range
        date_range = pd.date_range(
            start=daily_expenses['date'].min(),
            end=daily_expenses['date'].max(),
            freq='D'
        )
        
        # Reindex to include all dates (fill missing with 0)
        daily_expenses = daily_expenses.set_index('date').reindex(date_range, fill_value=0)
        daily_expenses.index.name = 'date'
        daily_expenses = daily_expenses.reset_index()
        
        return daily_expenses
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for machine learning models.
        
        Args:
            df: Time series dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df_features = df.copy()
        
        if 'date' not in df_features.columns:
            return df_features
        
        # Date-based features
        df_features['year'] = df_features['date'].dt.year
        df_features['month'] = df_features['date'].dt.month
        df_features['day'] = df_features['date'].dt.day
        df_features['weekday'] = df_features['date'].dt.dayofweek
        df_features['is_weekend'] = df_features['weekday'].isin([5, 6]).astype(int)
        df_features['day_of_year'] = df_features['date'].dt.dayofyear
        
        # Cyclical features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['weekday_sin'] = np.sin(2 * np.pi * df_features['weekday'] / 7)
        df_features['weekday_cos'] = np.cos(2 * np.pi * df_features['weekday'] / 7)
        
        # Lag features
        if 'abs_amount' in df_features.columns:
            for lag in [1, 7, 30]:
                df_features[f'lag_{lag}'] = df_features['abs_amount'].shift(lag)
            
            # Rolling statistics
            for window in [7, 30]:
                df_features[f'rolling_mean_{window}'] = df_features['abs_amount'].rolling(window=window).mean()
                df_features[f'rolling_std_{window}'] = df_features['abs_amount'].rolling(window=window).std()
        
        return df_features
    
    def train_linear_regression(self, df: pd.DataFrame) -> Dict:
        """Train linear regression model for forecasting."""
        df_features = self.create_features(df)
        
        # Use only basic features to avoid lag feature issues
        feature_cols = [
            'year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
        ]
        
        # Remove rows with NaN values
        df_clean = df_features.dropna(subset=['abs_amount'])
        
        if len(df_clean) < 10:  # Need minimum data points
            return {'model': None, 'error': 'Insufficient data for training'}
        
        # Only use available features
        available_features = [col for col in feature_cols if col in df_clean.columns]
        X = df_clean[available_features]
        y = df_clean['abs_amount']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate error metrics
        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Store model, scaler, and feature list
        self.models['linear_regression'] = model
        self.scalers['linear_regression'] = scaler
        self.models['linear_regression_features'] = available_features
        
        return {
            'model': model,
            'scaler': scaler,
            'features': available_features,
            'mae': mae,
            'rmse': rmse,
            'training_samples': len(df_clean)
        }
    
    def train_random_forest(self, df: pd.DataFrame) -> Dict:
        """Train random forest model for forecasting."""
        df_features = self.create_features(df)
        
        # Use only basic features to avoid lag feature issues
        feature_cols = [
            'year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
        ]
        
        # Remove rows with NaN values
        df_clean = df_features.dropna(subset=['abs_amount'])
        
        if len(df_clean) < 10:
            return {'model': None, 'error': 'Insufficient data for training'}
        
        # Only use available features
        available_features = [col for col in feature_cols if col in df_clean.columns]
        X = df_clean[available_features]
        y = df_clean['abs_amount']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate error metrics
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(available_features, model.feature_importances_))
        
        # Store model and feature list
        self.models['random_forest'] = model
        self.models['random_forest_features'] = available_features
        
        return {
            'model': model,
            'features': available_features,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': feature_importance,
            'training_samples': len(df_clean)
        }
    
    def train_arima_model(self, df: pd.DataFrame) -> Dict:
        """Train ARIMA model for time series forecasting."""
        if not STATSMODELS_AVAILABLE:
            return {'model': None, 'error': 'statsmodels not available'}
        
        if 'abs_amount' not in df.columns or len(df) < 30:
            return {'model': None, 'error': 'Insufficient data for ARIMA'}
        
        try:
            # Prepare time series
            ts = df.set_index('date')['abs_amount']
            
            # Auto ARIMA (simple approach)
            model = ARIMA(ts, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Calculate error metrics
            y_pred = fitted_model.fittedvalues
            y_true = ts[1:]  # ARIMA starts from second observation
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Store model
            self.models['arima'] = fitted_model
            
            return {
                'model': fitted_model,
                'mae': mae,
                'rmse': rmse,
                'aic': fitted_model.aic,
                'training_samples': len(ts)
            }
        
        except Exception as e:
            return {'model': None, 'error': f'ARIMA training failed: {str(e)}'}
    
    def forecast_next_period(self, df: pd.DataFrame, days: int = 30) -> Dict:
        """
        Generate forecasts for the next period using all available models.
        
        Args:
            df: Historical transaction data
            days: Number of days to forecast
            
        Returns:
            dict: Forecasting results from different models
        """
        # Prepare data
        time_series_data = self.prepare_time_series_data(df)
        
        if time_series_data.empty:
            return {'error': 'No data available for forecasting'}
        
        results = {}
        
        # Train and forecast with Linear Regression
        lr_result = self.train_linear_regression(time_series_data)
        if lr_result.get('model') is not None:
            lr_forecast = self._forecast_with_lr(time_series_data, days)
            results['linear_regression'] = {
                'training_metrics': lr_result,
                'forecast': lr_forecast
            }
        
        # Train and forecast with Random Forest
        rf_result = self.train_random_forest(time_series_data)
        if rf_result.get('model') is not None:
            rf_forecast = self._forecast_with_rf(time_series_data, days)
            results['random_forest'] = {
                'training_metrics': rf_result,
                'forecast': rf_forecast
            }
        
        # Train and forecast with ARIMA
        arima_result = self.train_arima_model(time_series_data)
        if arima_result.get('model') is not None:
            arima_forecast = self._forecast_with_arima(days)
            results['arima'] = {
                'training_metrics': arima_result,
                'forecast': arima_forecast
            }
        
        # Generate ensemble forecast
        if len(results) > 1:
            ensemble_forecast = self._create_ensemble_forecast(results, days)
            results['ensemble'] = {'forecast': ensemble_forecast}
        
        return results
    
    def _forecast_with_lr(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Generate forecast using Linear Regression."""
        model = self.models.get('linear_regression')
        scaler = self.scalers.get('linear_regression')
        
        if model is None or scaler is None:
            return pd.DataFrame()
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        # Create features for future dates
        future_df = pd.DataFrame({'date': future_dates})
        future_features = self.create_features(future_df)
        
        # Get the exact feature columns used during training
        training_result = self.models.get('linear_regression_features', [])
        if not training_result:
            # Fallback to basic features only
            feature_cols = [
                'year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year',
                'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
            ]
        else:
            feature_cols = training_result
        
        # Fill missing lag features with recent averages
        if 'abs_amount' in df.columns:
            recent_avg = df['abs_amount'].tail(7).mean()
            for col in feature_cols:
                if col.startswith('lag_') or col.startswith('rolling_'):
                    if col not in future_features.columns:
                        future_features[col] = recent_avg
                    else:
                        future_features[col] = future_features[col].fillna(recent_avg)
        
        # Select only the features that were used during training
        available_features = [col for col in feature_cols if col in future_features.columns]
        X_future = future_features[available_features].fillna(0)
        
        # Ensure we have the right number of features
        if X_future.shape[1] != scaler.n_features_in_:
            # Use only basic features if there's a mismatch
            basic_features = [
                'year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year',
                'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
            ]
            available_basic = [col for col in basic_features if col in future_features.columns]
            X_future = future_features[available_basic].fillna(0)
        
        try:
            X_future_scaled = scaler.transform(X_future)
            predictions = model.predict(X_future_scaled)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
        except Exception:
            # Fallback to simple average if prediction fails
            predictions = np.full(days, df['abs_amount'].mean() if 'abs_amount' in df.columns else 1000)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_amount': predictions
        })
        
        return forecast_df
    
    def _forecast_with_rf(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Generate forecast using Random Forest."""
        model = self.models.get('random_forest')
        
        if model is None:
            return pd.DataFrame()
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        # Create features for future dates
        future_df = pd.DataFrame({'date': future_dates})
        future_features = self.create_features(future_df)
        
        # Get the exact feature columns used during training
        training_features = self.models.get('random_forest_features', [])
        if not training_features:
            # Fallback to basic features only
            feature_cols = [
                'year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year',
                'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
            ]
        else:
            feature_cols = training_features
        
        # Only use available features
        available_features = [col for col in feature_cols if col in future_features.columns]
        X_future = future_features[available_features].fillna(0)
        
        try:
            predictions = model.predict(X_future)
            predictions = np.maximum(predictions, 0)
        except Exception:
            # Fallback to simple average if prediction fails
            predictions = np.full(days, df['abs_amount'].mean() if 'abs_amount' in df.columns else 1000)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_amount': predictions
        })
        
        return forecast_df
    
    def _forecast_with_arima(self, days: int) -> pd.DataFrame:
        """Generate forecast using ARIMA."""
        model = self.models.get('arima')
        
        if model is None:
            return pd.DataFrame()
        
        try:
            # Generate forecast
            forecast = model.forecast(steps=days)
            forecast = np.maximum(forecast, 0)  # Ensure non-negative
            
            # Create future dates
            last_date = model.data.dates[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_amount': forecast
            })
            
            return forecast_df
        
        except Exception:
            return pd.DataFrame()
    
    def _create_ensemble_forecast(self, results: Dict, days: int) -> pd.DataFrame:
        """Create ensemble forecast by averaging predictions from multiple models."""
        forecasts = []
        
        for model_name, result in results.items():
            if 'forecast' in result and not result['forecast'].empty:
                forecast_df = result['forecast'].copy()
                forecast_df['model'] = model_name
                forecasts.append(forecast_df)
        
        if not forecasts:
            return pd.DataFrame()
        
        # Combine all forecasts
        combined_df = pd.concat(forecasts, ignore_index=True)
        
        # Calculate ensemble average
        ensemble_df = combined_df.groupby('date')['predicted_amount'].mean().reset_index()
        ensemble_df['model'] = 'ensemble'
        
        return ensemble_df
    
    def get_monthly_forecast_summary(self, forecast_results: Dict) -> Dict:
        """Generate monthly summary from daily forecasts."""
        summary = {}
        
        for model_name, result in forecast_results.items():
            if 'forecast' in result and not result['forecast'].empty:
                forecast_df = result['forecast']
                
                # Group by month
                forecast_df['year_month'] = forecast_df['date'].dt.to_period('M')
                monthly_summary = forecast_df.groupby('year_month')['predicted_amount'].sum()
                
                summary[model_name] = {
                    'total_predicted': forecast_df['predicted_amount'].sum(),
                    'daily_average': forecast_df['predicted_amount'].mean(),
                    'monthly_breakdown': monthly_summary.to_dict()
                }
        
        return summary
