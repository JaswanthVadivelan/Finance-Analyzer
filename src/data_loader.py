"""
Data loader module for handling CSV and Excel bank statement uploads.
"""

import pandas as pd
import streamlit as st
from typing import Optional, Union
import io


class DataLoader:
    """Handles loading and initial validation of bank statement files."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file (CSV or Excel).
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame or None if loading fails
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def _load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV file with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with error handling
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
    
    def _load_excel(self, uploaded_file) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(uploaded_file)
    
    def validate_columns(self, df: pd.DataFrame) -> dict:
        """
        Validate if the dataframe has required columns for bank statements.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Validation results with suggested column mappings
        """
        required_fields = ['date', 'description', 'amount']
        column_mapping = {}
        validation_results = {
            'is_valid': True,
            'missing_fields': [],
            'suggested_mapping': {},
            'columns_found': list(df.columns)
        }
        
        # Common column name variations
        date_variations = ['date', 'transaction_date', 'trans_date', 'posting_date', 'value_date']
        desc_variations = ['description', 'transaction_description', 'details', 'particulars', 'narration']
        amount_variations = ['amount', 'transaction_amount', 'debit', 'credit', 'withdrawal', 'deposit']
        
        # Try to map columns
        df_columns_lower = [col.lower() for col in df.columns]
        
        # Map date column
        date_col = self._find_best_match(df_columns_lower, date_variations)
        if date_col:
            validation_results['suggested_mapping']['date'] = df.columns[df_columns_lower.index(date_col)]
        else:
            validation_results['missing_fields'].append('date')
            validation_results['is_valid'] = False
        
        # Map description column
        desc_col = self._find_best_match(df_columns_lower, desc_variations)
        if desc_col:
            validation_results['suggested_mapping']['description'] = df.columns[df_columns_lower.index(desc_col)]
        else:
            validation_results['missing_fields'].append('description')
            validation_results['is_valid'] = False
        
        # Map amount column
        amount_col = self._find_best_match(df_columns_lower, amount_variations)
        if amount_col:
            validation_results['suggested_mapping']['amount'] = df.columns[df_columns_lower.index(amount_col)]
        else:
            validation_results['missing_fields'].append('amount')
            validation_results['is_valid'] = False
        
        return validation_results
    
    def _find_best_match(self, columns: list, variations: list) -> Optional[str]:
        """Find the best matching column name from variations."""
        for variation in variations:
            if variation in columns:
                return variation
        
        # Try partial matches
        for variation in variations:
            for col in columns:
                if variation in col or col in variation:
                    return col
        
        return None
    
    def apply_column_mapping(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """
        Apply column mapping to standardize column names.
        
        Args:
            df: Input dataframe
            mapping: Dictionary mapping standard names to actual column names
            
        Returns:
            pd.DataFrame: Dataframe with standardized column names
        """
        df_mapped = df.copy()
        
        # Rename columns according to mapping
        rename_dict = {v: k for k, v in mapping.items()}
        df_mapped = df_mapped.rename(columns=rename_dict)
        
        return df_mapped
    
    def get_sample_data(self) -> pd.DataFrame:
        """Generate sample bank statement data for testing."""
        sample_data = {
            'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
            'description': ['SWIGGY BANGALORE', 'UBER TRIP', 'AMAZON PURCHASE', 'SALARY CREDIT', 'RENT PAYMENT'],
            'amount': [-450.50, -280.00, -1250.75, 50000.00, -15000.00]
        }
        return pd.DataFrame(sample_data)
