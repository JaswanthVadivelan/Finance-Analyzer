"""
Data cleaning module for standardizing and preprocessing bank statement data.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Optional, Union


class DataCleaner:
    """Handles cleaning and standardization of bank statement data."""
    
    def __init__(self):
        self.date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%d.%m.%Y', '%Y.%m.%d',
            '%d %b %Y', '%d %B %Y',
            '%b %d, %Y', '%B %d, %Y'
        ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning function that applies all cleaning steps.
        
        Args:
            df: Raw dataframe with bank statement data
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Clean dates
        df_clean = self.clean_dates(df_clean)
        
        # Clean amounts
        df_clean = self.clean_amounts(df_clean)
        
        # Clean descriptions
        df_clean = self.clean_descriptions(df_clean)
        
        # Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Sort by date
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        return df_clean
    
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize date column."""
        df_clean = df.copy()
        
        if 'date' not in df_clean.columns:
            return df_clean
        
        # Convert to string first
        df_clean['date'] = df_clean['date'].astype(str)
        
        # Try to parse dates with multiple formats
        parsed_dates = []
        for date_str in df_clean['date']:
            parsed_date = self._parse_date(date_str)
            parsed_dates.append(parsed_date)
        
        df_clean['date'] = parsed_dates
        
        # Remove rows with invalid dates
        df_clean = df_clean.dropna(subset=['date'])
        
        return df_clean
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string using multiple formats."""
        if pd.isna(date_str) or date_str.lower() in ['nan', 'none', '']:
            return None
        
        # Clean the date string
        date_str = str(date_str).strip()
        
        # Try each format
        for fmt in self.date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue
        
        # Try pandas automatic parsing as last resort
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except:
            return None
    
    def clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize amount column."""
        df_clean = df.copy()
        
        if 'amount' not in df_clean.columns:
            return df_clean
        
        # Convert to string for cleaning
        df_clean['amount_str'] = df_clean['amount'].astype(str)
        
        # Clean amount strings
        cleaned_amounts = []
        for amount_str in df_clean['amount_str']:
            cleaned_amount = self._clean_amount_string(amount_str)
            cleaned_amounts.append(cleaned_amount)
        
        df_clean['amount'] = cleaned_amounts
        
        # Convert to numeric
        df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
        
        # Drop the temporary column
        df_clean = df_clean.drop('amount_str', axis=1)
        
        return df_clean
    
    def _clean_amount_string(self, amount_str: str) -> Optional[float]:
        """Clean individual amount string."""
        if pd.isna(amount_str) or amount_str.lower() in ['nan', 'none', '']:
            return None
        
        # Convert to string and clean
        amount_str = str(amount_str).strip()
        
        # Remove currency symbols and common characters
        amount_str = re.sub(r'[₹$€£¥,\s]', '', amount_str)
        
        # Handle parentheses (usually indicate negative amounts)
        if '(' in amount_str and ')' in amount_str:
            amount_str = amount_str.replace('(', '').replace(')', '')
            amount_str = '-' + amount_str
        
        # Handle 'CR' and 'DR' suffixes
        if amount_str.upper().endswith('CR'):
            amount_str = amount_str[:-2].strip()
        elif amount_str.upper().endswith('DR'):
            amount_str = '-' + amount_str[:-2].strip()
        
        # Try to convert to float
        try:
            return float(amount_str)
        except (ValueError, TypeError):
            return None
    
    def clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize description column."""
        df_clean = df.copy()
        
        if 'description' not in df_clean.columns:
            return df_clean
        
        # Convert to string and clean
        df_clean['description'] = df_clean['description'].astype(str)
        df_clean['description'] = df_clean['description'].str.strip()
        df_clean['description'] = df_clean['description'].str.upper()
        
        # Remove extra whitespace
        df_clean['description'] = df_clean['description'].str.replace(r'\s+', ' ', regex=True)
        
        # Remove common prefixes/suffixes that don't add value
        patterns_to_remove = [
            r'^UPI-',
            r'^NEFT-',
            r'^IMPS-',
            r'^RTGS-',
            r'-\d{12}$',  # Remove transaction IDs at the end
            r'REF NO \d+',
        ]
        
        for pattern in patterns_to_remove:
            df_clean['description'] = df_clean['description'].str.replace(pattern, '', regex=True)
        
        df_clean['description'] = df_clean['description'].str.strip()
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate transactions."""
        df_clean = df.copy()
        
        # Define columns to check for duplicates
        duplicate_cols = ['date', 'description', 'amount']
        available_cols = [col for col in duplicate_cols if col in df_clean.columns]
        
        if len(available_cols) >= 2:
            df_clean = df_clean.drop_duplicates(subset=available_cols, keep='first')
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_clean = df.copy()
        
        # Remove rows where essential columns are missing
        essential_cols = ['date', 'amount']
        for col in essential_cols:
            if col in df_clean.columns:
                df_clean = df_clean.dropna(subset=[col])
        
        # Fill missing descriptions with placeholder
        if 'description' in df_clean.columns:
            df_clean['description'] = df_clean['description'].fillna('UNKNOWN TRANSACTION')
        
        return df_clean
    
    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns."""
        df_clean = df.copy()
        
        if 'date' in df_clean.columns:
            # Add date components
            df_clean['year'] = df_clean['date'].dt.year
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['day'] = df_clean['date'].dt.day
            df_clean['weekday'] = df_clean['date'].dt.day_name()
            df_clean['month_name'] = df_clean['date'].dt.month_name()
        
        if 'amount' in df_clean.columns:
            # Add transaction type
            df_clean['transaction_type'] = df_clean['amount'].apply(
                lambda x: 'Credit' if x > 0 else 'Debit'
            )
            
            # Add absolute amount
            df_clean['abs_amount'] = df_clean['amount'].abs()
        
        return df_clean
    
    def get_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """Generate summary of cleaning operations."""
        summary = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'rows_removed': len(original_df) - len(cleaned_df),
            'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100,
            'date_range': None,
            'amount_range': None
        }
        
        if 'date' in cleaned_df.columns and not cleaned_df.empty:
            summary['date_range'] = {
                'start': cleaned_df['date'].min(),
                'end': cleaned_df['date'].max()
            }
        
        if 'amount' in cleaned_df.columns and not cleaned_df.empty:
            summary['amount_range'] = {
                'min': cleaned_df['amount'].min(),
                'max': cleaned_df['amount'].max()
            }
        
        return summary
