"""
Transaction categorization module using regex patterns and keyword matching.
"""

import pandas as pd
import re
from typing import Dict, List, Optional


class TransactionCategorizer:
    """Categorizes transactions based on description patterns and keywords."""
    
    def __init__(self):
        self.categories = {
            'Food & Dining': {
                'keywords': [
                    'swiggy', 'zomato', 'uber eats', 'dominos', 'pizza hut', 'mcdonalds', 'kfc',
                    'restaurant', 'cafe', 'food', 'dining', 'meal', 'lunch', 'dinner', 'breakfast',
                    'starbucks', 'costa coffee', 'cafe coffee day', 'ccd', 'dunkin', 'subway',
                    'burger king', 'taco bell', 'haldirams', 'barbeque nation'
                ],
                'patterns': [
                    r'.*food.*', r'.*restaurant.*', r'.*cafe.*', r'.*dining.*',
                    r'.*swiggy.*', r'.*zomato.*', r'.*uber\s*eats.*'
                ]
            },
            'Transportation': {
                'keywords': [
                    'uber', 'ola', 'taxi', 'cab', 'auto', 'metro', 'bus', 'train', 'flight',
                    'petrol', 'diesel', 'fuel', 'gas', 'parking', 'toll', 'rapido',
                    'indigo', 'spicejet', 'air india', 'vistara', 'irctc', 'redbus'
                ],
                'patterns': [
                    r'.*uber.*', r'.*ola.*', r'.*taxi.*', r'.*cab.*', r'.*metro.*',
                    r'.*petrol.*', r'.*fuel.*', r'.*parking.*', r'.*toll.*'
                ]
            },
            'Shopping': {
                'keywords': [
                    'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'shopping', 'mall',
                    'store', 'market', 'supermarket', 'grocery', 'big bazaar', 'reliance',
                    'dmart', 'more', 'spencer', 'lifestyle', 'pantaloons', 'westside',
                    'max fashion', 'h&m', 'zara', 'uniqlo'
                ],
                'patterns': [
                    r'.*amazon.*', r'.*flipkart.*', r'.*myntra.*', r'.*shopping.*',
                    r'.*mall.*', r'.*store.*', r'.*market.*'
                ]
            },
            'Bills & Utilities': {
                'keywords': [
                    'electricity', 'water', 'gas', 'internet', 'wifi', 'mobile', 'phone',
                    'recharge', 'bill', 'utility', 'bsnl', 'airtel', 'jio', 'vodafone',
                    'idea', 'broadband', 'cable', 'dish tv', 'tata sky', 'sun direct'
                ],
                'patterns': [
                    r'.*bill.*', r'.*utility.*', r'.*electricity.*', r'.*water.*',
                    r'.*recharge.*', r'.*mobile.*', r'.*internet.*'
                ]
            },
            'Entertainment': {
                'keywords': [
                    'netflix', 'amazon prime', 'hotstar', 'zee5', 'sony liv', 'voot',
                    'youtube', 'spotify', 'gaana', 'jiosaavn', 'movie', 'cinema',
                    'theater', 'theatre', 'pvr', 'inox', 'entertainment', 'game',
                    'gaming', 'steam', 'playstation', 'xbox'
                ],
                'patterns': [
                    r'.*netflix.*', r'.*prime.*', r'.*hotstar.*', r'.*movie.*',
                    r'.*cinema.*', r'.*entertainment.*', r'.*gaming.*'
                ]
            },
            'Healthcare': {
                'keywords': [
                    'hospital', 'clinic', 'doctor', 'medical', 'pharmacy', 'medicine',
                    'health', 'apollo', 'fortis', 'max healthcare', 'aiims', 'medplus',
                    'pharmeasy', '1mg', 'netmeds', 'dental', 'eye care', 'lab test'
                ],
                'patterns': [
                    r'.*hospital.*', r'.*medical.*', r'.*pharmacy.*', r'.*health.*',
                    r'.*doctor.*', r'.*clinic.*'
                ]
            },
            'Education': {
                'keywords': [
                    'school', 'college', 'university', 'education', 'course', 'training',
                    'tuition', 'coaching', 'byju', 'unacademy', 'vedantu', 'coursera',
                    'udemy', 'book', 'library', 'exam', 'fee'
                ],
                'patterns': [
                    r'.*school.*', r'.*college.*', r'.*education.*', r'.*course.*',
                    r'.*training.*', r'.*tuition.*'
                ]
            },
            'Investment & Savings': {
                'keywords': [
                    'mutual fund', 'sip', 'investment', 'stock', 'share', 'equity',
                    'bond', 'fd', 'fixed deposit', 'rd', 'recurring deposit', 'ppf',
                    'nsc', 'insurance', 'lic', 'hdfc life', 'icici prudential',
                    'zerodha', 'groww', 'upstox', 'angel broking'
                ],
                'patterns': [
                    r'.*mutual.*fund.*', r'.*sip.*', r'.*investment.*', r'.*insurance.*',
                    r'.*zerodha.*', r'.*groww.*'
                ]
            },
            'Rent & Housing': {
                'keywords': [
                    'rent', 'house rent', 'flat rent', 'apartment', 'housing',
                    'maintenance', 'society', 'electricity deposit', 'security deposit'
                ],
                'patterns': [
                    r'.*rent.*', r'.*housing.*', r'.*maintenance.*', r'.*society.*'
                ]
            },
            'Personal Care': {
                'keywords': [
                    'salon', 'spa', 'beauty', 'haircut', 'parlour', 'cosmetics',
                    'skincare', 'grooming', 'lakme', 'vlcc', 'naturals'
                ],
                'patterns': [
                    r'.*salon.*', r'.*beauty.*', r'.*spa.*', r'.*cosmetics.*'
                ]
            },
            'Income': {
                'keywords': [
                    'salary', 'wages', 'income', 'bonus', 'incentive', 'refund',
                    'cashback', 'interest', 'dividend', 'freelance', 'consulting'
                ],
                'patterns': [
                    r'.*salary.*', r'.*wages.*', r'.*bonus.*', r'.*refund.*',
                    r'.*cashback.*', r'.*interest.*'
                ]
            },
            'ATM & Banking': {
                'keywords': [
                    'atm', 'cash withdrawal', 'bank charges', 'service charge',
                    'annual fee', 'processing fee', 'transaction charge'
                ],
                'patterns': [
                    r'.*atm.*', r'.*withdrawal.*', r'.*bank.*charge.*', r'.*fee.*'
                ]
            }
        }
        
        # Compile regex patterns for better performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        for category in self.categories:
            compiled_patterns = []
            for pattern in self.categories[category]['patterns']:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            self.categories[category]['compiled_patterns'] = compiled_patterns
    
    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize all transactions in the dataframe.
        
        Args:
            df: Dataframe with transaction data
            
        Returns:
            pd.DataFrame: Dataframe with added 'category' column
        """
        df_categorized = df.copy()
        
        if 'description' not in df_categorized.columns:
            df_categorized['category'] = 'Other'
            return df_categorized
        
        # Apply categorization
        categories = []
        for description in df_categorized['description']:
            category = self.categorize_single_transaction(description)
            categories.append(category)
        
        df_categorized['category'] = categories
        
        return df_categorized
    
    def categorize_single_transaction(self, description: str) -> str:
        """
        Categorize a single transaction based on its description.
        
        Args:
            description: Transaction description
            
        Returns:
            str: Category name
        """
        if pd.isna(description) or description == '':
            return 'Other'
        
        description = str(description).lower().strip()
        
        # Check each category
        for category_name, category_data in self.categories.items():
            # Check keywords
            for keyword in category_data['keywords']:
                if keyword.lower() in description:
                    return category_name
            
            # Check regex patterns
            for pattern in category_data['compiled_patterns']:
                if pattern.search(description):
                    return category_name
        
        return 'Other'
    
    def get_category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each category.
        
        Args:
            df: Dataframe with categorized transactions
            
        Returns:
            pd.DataFrame: Summary statistics by category
        """
        if 'category' not in df.columns or 'amount' not in df.columns:
            return pd.DataFrame()
        
        # Filter only debit transactions for expense analysis
        expense_df = df[df['amount'] < 0].copy()
        expense_df['abs_amount'] = expense_df['amount'].abs()
        
        # Calculate summary statistics
        summary = expense_df.groupby('category').agg({
            'abs_amount': ['count', 'sum', 'mean', 'median'],
            'amount': 'sum'
        }).round(2)
        
        # Flatten column names
        summary.columns = ['transaction_count', 'total_spent', 'avg_amount', 'median_amount', 'net_amount']
        
        # Calculate percentage of total spending
        total_spending = summary['total_spent'].sum()
        summary['percentage'] = (summary['total_spent'] / total_spending * 100).round(2)
        
        # Sort by total spending
        summary = summary.sort_values('total_spent', ascending=False)
        
        return summary.reset_index()
    
    def get_monthly_category_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get monthly spending trends by category.
        
        Args:
            df: Dataframe with categorized transactions
            
        Returns:
            pd.DataFrame: Monthly category trends
        """
        if not all(col in df.columns for col in ['category', 'amount', 'date']):
            return pd.DataFrame()
        
        # Filter only debit transactions
        expense_df = df[df['amount'] < 0].copy()
        expense_df['abs_amount'] = expense_df['amount'].abs()
        expense_df['year_month'] = expense_df['date'].dt.to_period('M')
        
        # Group by month and category
        monthly_trends = expense_df.groupby(['year_month', 'category'])['abs_amount'].sum().reset_index()
        
        # Pivot to get categories as columns
        monthly_pivot = monthly_trends.pivot(index='year_month', columns='category', values='abs_amount').fillna(0)
        
        return monthly_pivot.reset_index()
    
    def add_custom_category(self, category_name: str, keywords: List[str], patterns: List[str] = None):
        """
        Add a custom category with keywords and patterns.
        
        Args:
            category_name: Name of the new category
            keywords: List of keywords to match
            patterns: List of regex patterns to match
        """
        if patterns is None:
            patterns = []
        
        self.categories[category_name] = {
            'keywords': keywords,
            'patterns': patterns
        }
        
        # Compile new patterns
        compiled_patterns = []
        for pattern in patterns:
            compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
        self.categories[category_name]['compiled_patterns'] = compiled_patterns
    
    def get_uncategorized_transactions(self, df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
        """
        Get transactions that were categorized as 'Other' for manual review.
        
        Args:
            df: Dataframe with categorized transactions
            limit: Maximum number of transactions to return
            
        Returns:
            pd.DataFrame: Uncategorized transactions
        """
        if 'category' not in df.columns:
            return pd.DataFrame()
        
        uncategorized = df[df['category'] == 'Other'].copy()
        
        if len(uncategorized) > limit:
            # Return highest value transactions first
            if 'amount' in uncategorized.columns:
                uncategorized = uncategorized.reindex(uncategorized['amount'].abs().sort_values(ascending=False).index)
            uncategorized = uncategorized.head(limit)
        
        return uncategorized[['date', 'description', 'amount']].reset_index(drop=True)
