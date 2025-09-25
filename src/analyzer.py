"""
Financial analysis module for generating insights and trends from transaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class FinancialAnalyzer:
    """Analyzes transaction data to generate financial insights and trends."""
    
    def __init__(self):
        pass
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive financial insights from transaction data.
        
        Args:
            df: Dataframe with cleaned and categorized transactions
            
        Returns:
            dict: Dictionary containing various financial insights
        """
        insights = {
            'overview': self.get_overview_stats(df),
            'spending_by_category': self.get_spending_by_category(df),
            'monthly_trends': self.get_monthly_trends(df),
            'weekly_patterns': self.get_weekly_patterns(df),
            'top_transactions': self.get_top_transactions(df),
            'income_vs_expenses': self.get_income_vs_expenses(df),
            'savings_rate': self.calculate_savings_rate(df),
            'spending_velocity': self.calculate_spending_velocity(df)
        }
        
        return insights
    
    def get_overview_stats(self, df: pd.DataFrame) -> Dict:
        """Get basic overview statistics."""
        if df.empty or 'amount' not in df.columns:
            return {}
        
        total_transactions = len(df)
        total_income = df[df['amount'] > 0]['amount'].sum() if any(df['amount'] > 0) else 0
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum()) if any(df['amount'] < 0) else 0
        net_flow = total_income - total_expenses
        
        avg_transaction = df['amount'].mean()
        largest_expense = abs(df[df['amount'] < 0]['amount'].min()) if any(df['amount'] < 0) else 0
        largest_income = df[df['amount'] > 0]['amount'].max() if any(df['amount'] > 0) else 0
        
        date_range = None
        if 'date' in df.columns and not df.empty:
            date_range = {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            }
        
        return {
            'total_transactions': total_transactions,
            'total_income': round(total_income, 2),
            'total_expenses': round(total_expenses, 2),
            'net_cash_flow': round(net_flow, 2),
            'avg_transaction_amount': round(avg_transaction, 2),
            'largest_expense': round(largest_expense, 2),
            'largest_income': round(largest_income, 2),
            'date_range': date_range
        }
    
    def get_spending_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get spending breakdown by category."""
        if 'category' not in df.columns or 'amount' not in df.columns:
            return pd.DataFrame()
        
        # Filter only expenses (negative amounts)
        expenses_df = df[df['amount'] < 0].copy()
        expenses_df['abs_amount'] = expenses_df['amount'].abs()
        
        category_spending = expenses_df.groupby('category').agg({
            'abs_amount': ['sum', 'count', 'mean'],
            'amount': 'sum'
        }).round(2)
        
        # Flatten column names
        category_spending.columns = ['total_spent', 'transaction_count', 'avg_amount', 'net_amount']
        
        # Calculate percentage
        total_spending = category_spending['total_spent'].sum()
        category_spending['percentage'] = (category_spending['total_spent'] / total_spending * 100).round(2)
        
        # Sort by spending amount
        category_spending = category_spending.sort_values('total_spent', ascending=False)
        
        return category_spending.reset_index()
    
    def get_monthly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get monthly spending and income trends."""
        if 'date' not in df.columns or 'amount' not in df.columns:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy['year_month'] = df_copy['date'].dt.to_period('M')
        
        monthly_stats = df_copy.groupby('year_month').agg({
            'amount': ['sum', 'count'],
        }).round(2)
        
        # Flatten column names
        monthly_stats.columns = ['net_amount', 'transaction_count']
        
        # Calculate income and expenses separately
        income_monthly = df_copy[df_copy['amount'] > 0].groupby('year_month')['amount'].sum()
        expenses_monthly = df_copy[df_copy['amount'] < 0].groupby('year_month')['amount'].sum().abs()
        
        monthly_stats['income'] = income_monthly
        monthly_stats['expenses'] = expenses_monthly
        monthly_stats['savings'] = monthly_stats['income'] - monthly_stats['expenses']
        
        # Fill NaN values with 0
        monthly_stats = monthly_stats.fillna(0)
        
        return monthly_stats.reset_index()
    
    def get_weekly_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze spending patterns by day of week."""
        if 'date' not in df.columns or 'amount' not in df.columns:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy['weekday'] = df_copy['date'].dt.day_name()
        df_copy['weekday_num'] = df_copy['date'].dt.dayofweek
        
        # Focus on expenses
        expenses_df = df_copy[df_copy['amount'] < 0].copy()
        expenses_df['abs_amount'] = expenses_df['amount'].abs()
        
        weekly_patterns = expenses_df.groupby(['weekday', 'weekday_num']).agg({
            'abs_amount': ['sum', 'count', 'mean']
        }).round(2)
        
        # Flatten column names
        weekly_patterns.columns = ['total_spent', 'transaction_count', 'avg_amount']
        
        # Sort by weekday number (Monday = 0)
        weekly_patterns = weekly_patterns.reset_index().sort_values('weekday_num')
        
        return weekly_patterns[['weekday', 'total_spent', 'transaction_count', 'avg_amount']]
    
    def get_top_transactions(self, df: pd.DataFrame, limit: int = 10) -> Dict:
        """Get top transactions by amount (both income and expenses)."""
        if 'amount' not in df.columns:
            return {'top_expenses': pd.DataFrame(), 'top_income': pd.DataFrame()}
        
        # Top expenses (most negative amounts)
        expenses = df[df['amount'] < 0].copy()
        expenses['abs_amount'] = expenses['amount'].abs()
        top_expenses = expenses.nlargest(limit, 'abs_amount')[['date', 'description', 'amount', 'category']]
        
        # Top income (most positive amounts)
        income = df[df['amount'] > 0].copy()
        top_income = income.nlargest(limit, 'amount')[['date', 'description', 'amount', 'category']]
        
        return {
            'top_expenses': top_expenses.reset_index(drop=True),
            'top_income': top_income.reset_index(drop=True)
        }
    
    def get_income_vs_expenses(self, df: pd.DataFrame) -> Dict:
        """Compare income vs expenses over time."""
        if 'amount' not in df.columns:
            return {}
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        income_transactions = len(df[df['amount'] > 0])
        expense_transactions = len(df[df['amount'] < 0])
        
        avg_income_per_transaction = total_income / income_transactions if income_transactions > 0 else 0
        avg_expense_per_transaction = total_expenses / expense_transactions if expense_transactions > 0 else 0
        
        return {
            'total_income': round(total_income, 2),
            'total_expenses': round(total_expenses, 2),
            'net_savings': round(total_income - total_expenses, 2),
            'income_transactions': income_transactions,
            'expense_transactions': expense_transactions,
            'avg_income_per_transaction': round(avg_income_per_transaction, 2),
            'avg_expense_per_transaction': round(avg_expense_per_transaction, 2),
            'expense_to_income_ratio': round(total_expenses / total_income, 2) if total_income > 0 else 0
        }
    
    def calculate_savings_rate(self, df: pd.DataFrame) -> Dict:
        """Calculate savings rate and related metrics."""
        if 'amount' not in df.columns:
            return {}
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        if total_income <= 0:
            return {'savings_rate': 0, 'monthly_savings_rate': {}}
        
        overall_savings_rate = ((total_income - total_expenses) / total_income) * 100
        
        # Monthly savings rate
        monthly_savings = {}
        if 'date' in df.columns:
            df_copy = df.copy()
            df_copy['year_month'] = df_copy['date'].dt.to_period('M')
            
            for month in df_copy['year_month'].unique():
                month_data = df_copy[df_copy['year_month'] == month]
                month_income = month_data[month_data['amount'] > 0]['amount'].sum()
                month_expenses = abs(month_data[month_data['amount'] < 0]['amount'].sum())
                
                if month_income > 0:
                    month_savings_rate = ((month_income - month_expenses) / month_income) * 100
                    monthly_savings[str(month)] = round(month_savings_rate, 2)
        
        return {
            'overall_savings_rate': round(overall_savings_rate, 2),
            'monthly_savings_rate': monthly_savings,
            'target_savings_rate': 20,  # Recommended 20% savings rate
            'meets_target': overall_savings_rate >= 20
        }
    
    def calculate_spending_velocity(self, df: pd.DataFrame) -> Dict:
        """Calculate how quickly money is being spent."""
        if 'date' not in df.columns or 'amount' not in df.columns:
            return {}
        
        expenses_df = df[df['amount'] < 0].copy()
        if expenses_df.empty:
            return {}
        
        expenses_df = expenses_df.sort_values('date')
        expenses_df['abs_amount'] = expenses_df['amount'].abs()
        
        # Calculate daily spending
        daily_spending = expenses_df.groupby(expenses_df['date'].dt.date)['abs_amount'].sum()
        
        # Calculate metrics
        avg_daily_spending = daily_spending.mean()
        max_daily_spending = daily_spending.max()
        days_with_spending = len(daily_spending)
        
        # Calculate spending streaks (consecutive days with spending)
        spending_days = set(daily_spending.index)
        all_days = pd.date_range(start=daily_spending.index.min(), end=daily_spending.index.max()).date
        
        current_streak = 0
        max_streak = 0
        temp_streak = 0
        
        for day in all_days:
            if day in spending_days:
                temp_streak += 1
                max_streak = max(max_streak, temp_streak)
            else:
                temp_streak = 0
        
        # Current streak (from the end)
        for day in reversed(all_days):
            if day in spending_days:
                current_streak += 1
            else:
                break
        
        return {
            'avg_daily_spending': round(avg_daily_spending, 2),
            'max_daily_spending': round(max_daily_spending, 2),
            'days_with_spending': days_with_spending,
            'current_spending_streak': current_streak,
            'max_spending_streak': max_streak,
            'spending_frequency': round((days_with_spending / len(all_days)) * 100, 2)
        }
    
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect unusual transactions based on statistical analysis."""
        if 'amount' not in df.columns:
            return pd.DataFrame()
        
        expenses_df = df[df['amount'] < 0].copy()
        expenses_df['abs_amount'] = expenses_df['amount'].abs()
        
        # Calculate z-scores
        mean_amount = expenses_df['abs_amount'].mean()
        std_amount = expenses_df['abs_amount'].std()
        
        if std_amount == 0:
            return pd.DataFrame()
        
        expenses_df['z_score'] = (expenses_df['abs_amount'] - mean_amount) / std_amount
        
        # Find anomalies
        anomalies = expenses_df[abs(expenses_df['z_score']) > threshold].copy()
        anomalies = anomalies.sort_values('abs_amount', ascending=False)
        
        return anomalies[['date', 'description', 'amount', 'category', 'z_score']].reset_index(drop=True)
    
    def generate_insights_summary(self, insights: Dict) -> str:
        """Generate a text summary of key insights."""
        summary_parts = []
        
        # Overview
        overview = insights.get('overview', {})
        if overview:
            summary_parts.append(f"📊 **Financial Overview**")
            summary_parts.append(f"• Total transactions: {overview.get('total_transactions', 0)}")
            summary_parts.append(f"• Total income: ₹{overview.get('total_income', 0):,.2f}")
            summary_parts.append(f"• Total expenses: ₹{overview.get('total_expenses', 0):,.2f}")
            summary_parts.append(f"• Net cash flow: ₹{overview.get('net_cash_flow', 0):,.2f}")
            summary_parts.append("")
        
        # Top spending categories
        spending_by_category = insights.get('spending_by_category')
        if not spending_by_category.empty:
            summary_parts.append(f"💰 **Top Spending Categories**")
            top_3_categories = spending_by_category.head(3)
            for _, row in top_3_categories.iterrows():
                summary_parts.append(f"• {row['category']}: ₹{row['total_spent']:,.2f} ({row['percentage']:.1f}%)")
            summary_parts.append("")
        
        # Savings rate
        savings_info = insights.get('savings_rate', {})
        if savings_info:
            rate = savings_info.get('overall_savings_rate', 0)
            meets_target = savings_info.get('meets_target', False)
            status = "✅" if meets_target else "⚠️"
            summary_parts.append(f"💡 **Savings Rate**: {rate:.1f}% {status}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
