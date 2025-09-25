"""
Utility functions for the Finance Analyzer application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Union


def format_currency(amount: float, currency_symbol: str = "₹") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency_symbol: Currency symbol to use
        
    Returns:
        str: Formatted currency string
    """
    return f"{currency_symbol}{amount:,.2f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value to format (e.g., 0.15 for 15%)
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        float: Result of division or default
    """
    return numerator / denominator if denominator != 0 else default


def clean_text(text: str) -> str:
    """
    Clean and standardize text.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and clean
    text = str(text).strip().upper()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def get_date_range_description(start_date: datetime, end_date: datetime) -> str:
    """
    Get human-readable description of date range.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        str: Description of date range
    """
    delta = end_date - start_date
    days = delta.days
    
    if days == 0:
        return "Same day"
    elif days == 1:
        return "1 day"
    elif days < 7:
        return f"{days} days"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    elif days < 365:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''}"
    else:
        years = days // 365
        return f"{years} year{'s' if years > 1 else ''}"


def calculate_growth_rate(current: float, previous: float) -> Optional[float]:
    """
    Calculate growth rate between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        float: Growth rate as percentage (or None if calculation not possible)
    """
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return None
    
    return ((current - previous) / abs(previous)) * 100


def get_financial_health_score(
    savings_rate: float,
    expense_ratio: float,
    transaction_regularity: float
) -> Dict[str, Union[float, str]]:
    """
    Calculate a simple financial health score.
    
    Args:
        savings_rate: Savings rate as percentage
        expense_ratio: Expense to income ratio
        transaction_regularity: Regularity of transactions (0-1)
        
    Returns:
        dict: Financial health score and interpretation
    """
    # Normalize scores (0-100)
    savings_score = min(savings_rate * 5, 100)  # 20% savings = 100 points
    expense_score = max(100 - (expense_ratio * 100), 0)  # Lower expense ratio = higher score
    regularity_score = transaction_regularity * 100
    
    # Weighted average
    overall_score = (savings_score * 0.4 + expense_score * 0.4 + regularity_score * 0.2)
    
    # Interpretation
    if overall_score >= 80:
        interpretation = "Excellent"
        emoji = "🟢"
    elif overall_score >= 60:
        interpretation = "Good"
        emoji = "🟡"
    elif overall_score >= 40:
        interpretation = "Fair"
        emoji = "🟠"
    else:
        interpretation = "Needs Improvement"
        emoji = "🔴"
    
    return {
        'score': round(overall_score, 1),
        'interpretation': interpretation,
        'emoji': emoji,
        'savings_score': round(savings_score, 1),
        'expense_score': round(expense_score, 1),
        'regularity_score': round(regularity_score, 1)
    }


def generate_insights_text(insights: Dict) -> List[str]:
    """
    Generate human-readable insights from analysis results.
    
    Args:
        insights: Dictionary containing analysis results
        
    Returns:
        list: List of insight strings
    """
    insights_list = []
    
    # Overview insights
    overview = insights.get('overview', {})
    if overview:
        total_transactions = overview.get('total_transactions', 0)
        net_flow = overview.get('net_cash_flow', 0)
        
        insights_list.append(f"📊 You have {total_transactions} transactions in your data")
        
        if net_flow > 0:
            insights_list.append(f"💰 You have a positive cash flow of {format_currency(net_flow)}")
        else:
            insights_list.append(f"⚠️ You have a negative cash flow of {format_currency(abs(net_flow))}")
    
    # Spending insights
    spending_by_category = insights.get('spending_by_category')
    if not spending_by_category.empty:
        top_category = spending_by_category.iloc[0]
        insights_list.append(
            f"🛍️ Your highest spending category is {top_category['category']} "
            f"({format_percentage(top_category['percentage'])} of total expenses)"
        )
        
        if len(spending_by_category) >= 3:
            top_3_percentage = spending_by_category.head(3)['percentage'].sum()
            insights_list.append(
                f"📈 Your top 3 categories account for {format_percentage(top_3_percentage)} of spending"
            )
    
    # Savings insights
    savings_info = insights.get('savings_rate', {})
    if savings_info:
        rate = savings_info.get('overall_savings_rate', 0)
        if rate >= 20:
            insights_list.append(f"✅ Great job! Your savings rate of {format_percentage(rate)} meets the recommended 20%")
        elif rate >= 10:
            insights_list.append(f"👍 Your savings rate of {format_percentage(rate)} is decent, try to reach 20%")
        else:
            insights_list.append(f"⚠️ Your savings rate of {format_percentage(rate)} is below recommended levels")
    
    # Weekly patterns
    weekly_patterns = insights.get('weekly_patterns')
    if not weekly_patterns.empty:
        highest_spending_day = weekly_patterns.loc[weekly_patterns['total_spent'].idxmax(), 'weekday']
        insights_list.append(f"📅 You spend the most on {highest_spending_day}s")
    
    return insights_list


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate if dataframe has required columns and data.
    
    Args:
        df: Dataframe to validate
        required_columns: List of required column names
        
    Returns:
        dict: Validation results
    """
    result = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'issues': []
    }
    
    # Check if dataframe is empty
    if df.empty:
        result['is_valid'] = False
        result['issues'].append('Dataframe is empty')
        return result
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        result['is_valid'] = False
        result['missing_columns'] = missing_cols
        result['issues'].append(f'Missing columns: {missing_cols}')
    
    # Check for empty columns
    for col in required_columns:
        if col in df.columns and df[col].isna().all():
            result['empty_columns'].append(col)
            result['issues'].append(f'Column {col} is completely empty')
    
    return result


def create_summary_stats(df: pd.DataFrame, amount_column: str = 'amount') -> Dict:
    """
    Create summary statistics for financial data.
    
    Args:
        df: Dataframe with financial data
        amount_column: Name of amount column
        
    Returns:
        dict: Summary statistics
    """
    if amount_column not in df.columns or df.empty:
        return {}
    
    amounts = df[amount_column]
    
    return {
        'count': len(amounts),
        'mean': amounts.mean(),
        'median': amounts.median(),
        'std': amounts.std(),
        'min': amounts.min(),
        'max': amounts.max(),
        'sum': amounts.sum(),
        'positive_count': (amounts > 0).sum(),
        'negative_count': (amounts < 0).sum(),
        'zero_count': (amounts == 0).sum()
    }


def export_insights_to_dict(insights: Dict) -> Dict:
    """
    Convert insights to a serializable dictionary for export.
    
    Args:
        insights: Insights dictionary containing DataFrames
        
    Returns:
        dict: Serializable insights dictionary
    """
    export_dict = {}
    
    for key, value in insights.items():
        if isinstance(value, pd.DataFrame):
            export_dict[key] = value.to_dict('records')
        elif isinstance(value, dict):
            export_dict[key] = value
        else:
            export_dict[key] = str(value)
    
    return export_dict


def get_color_palette(n_colors: int) -> List[str]:
    """
    Get a color palette for visualizations.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        list: List of color hex codes
    """
    # Default color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    # Repeat colors if needed
    while len(colors) < n_colors:
        colors.extend(colors)
    
    return colors[:n_colors]
