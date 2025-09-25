"""
Test script to verify all components of the Finance Analyzer work correctly.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        from src.data_loader import DataLoader
        from src.cleaner import DataCleaner
        from src.categorizer import TransactionCategorizer
        from src.analyzer import FinancialAnalyzer
        from src.forecaster import ExpenseForecaster
        from src.utils import format_currency, clean_text
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loader():
    """Test data loading functionality."""
    try:
        loader = DataLoader()
        
        # Test sample data generation
        sample_data = loader.get_sample_data()
        assert not sample_data.empty, "Sample data should not be empty"
        assert 'date' in sample_data.columns, "Sample data should have date column"
        assert 'description' in sample_data.columns, "Sample data should have description column"
        assert 'amount' in sample_data.columns, "Sample data should have amount column"
        
        print("✅ DataLoader tests passed")
        return True
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False

def test_data_cleaner():
    """Test data cleaning functionality."""
    try:
        cleaner = DataCleaner()
        loader = DataLoader()
        
        # Get sample data and clean it
        sample_data = loader.get_sample_data()
        cleaned_data = cleaner.clean_data(sample_data)
        
        assert not cleaned_data.empty, "Cleaned data should not be empty"
        assert 'date' in cleaned_data.columns, "Cleaned data should have date column"
        
        # Test derived columns
        enhanced_data = cleaner.add_derived_columns(cleaned_data)
        assert 'year' in enhanced_data.columns, "Should have year column"
        assert 'month' in enhanced_data.columns, "Should have month column"
        
        print("✅ DataCleaner tests passed")
        return True
    except Exception as e:
        print(f"❌ DataCleaner test failed: {e}")
        return False

def test_categorizer():
    """Test transaction categorization."""
    try:
        categorizer = TransactionCategorizer()
        
        # Test single transaction categorization
        test_descriptions = [
            "SWIGGY BANGALORE",
            "UBER TRIP",
            "AMAZON PURCHASE",
            "SALARY CREDIT",
            "RENT PAYMENT"
        ]
        
        for desc in test_descriptions:
            category = categorizer.categorize_single_transaction(desc)
            assert category is not None, f"Category should not be None for {desc}"
            assert isinstance(category, str), f"Category should be string for {desc}"
        
        # Test dataframe categorization
        loader = DataLoader()
        sample_data = loader.get_sample_data()
        categorized_data = categorizer.categorize_transactions(sample_data)
        
        assert 'category' in categorized_data.columns, "Should have category column"
        assert not categorized_data['category'].isna().all(), "Should have some categorized transactions"
        
        print("✅ TransactionCategorizer tests passed")
        return True
    except Exception as e:
        print(f"❌ TransactionCategorizer test failed: {e}")
        return False

def test_analyzer():
    """Test financial analysis functionality."""
    try:
        analyzer = FinancialAnalyzer()
        loader = DataLoader()
        cleaner = DataCleaner()
        categorizer = TransactionCategorizer()
        
        # Prepare test data
        sample_data = loader.get_sample_data()
        cleaned_data = cleaner.clean_data(sample_data)
        cleaned_data = cleaner.add_derived_columns(cleaned_data)
        categorized_data = categorizer.categorize_transactions(cleaned_data)
        
        # Generate insights
        insights = analyzer.generate_insights(categorized_data)
        
        assert isinstance(insights, dict), "Insights should be a dictionary"
        assert 'overview' in insights, "Should have overview insights"
        assert 'spending_by_category' in insights, "Should have category insights"
        
        print("✅ FinancialAnalyzer tests passed")
        return True
    except Exception as e:
        print(f"❌ FinancialAnalyzer test failed: {e}")
        return False

def test_forecaster():
    """Test expense forecasting functionality."""
    try:
        forecaster = ExpenseForecaster()
        loader = DataLoader()
        cleaner = DataCleaner()
        
        # Prepare test data
        sample_data = loader.get_sample_data()
        cleaned_data = cleaner.clean_data(sample_data)
        cleaned_data = cleaner.add_derived_columns(cleaned_data)
        
        # Test time series data preparation
        ts_data = forecaster.prepare_time_series_data(cleaned_data)
        assert not ts_data.empty, "Time series data should not be empty"
        
        print("✅ ExpenseForecaster tests passed")
        return True
    except Exception as e:
        print(f"❌ ExpenseForecaster test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    try:
        from src.utils import format_currency, clean_text, safe_divide
        
        # Test currency formatting
        formatted = format_currency(1234.56)
        assert "₹" in formatted, "Should contain currency symbol"
        assert "1,234.56" in formatted, "Should format number correctly"
        
        # Test text cleaning
        cleaned = clean_text("  test  text  ")
        assert cleaned == "TEST TEXT", "Should clean and uppercase text"
        
        # Test safe division
        result = safe_divide(10, 2)
        assert result == 5.0, "Should divide correctly"
        
        result = safe_divide(10, 0, default=0)
        assert result == 0, "Should return default for division by zero"
        
        print("✅ Utils tests passed")
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_sample_data_file():
    """Test that sample data file exists and is valid."""
    try:
        sample_file = os.path.join(os.path.dirname(__file__), 'data', 'sample_bank_statement.csv')
        
        if not os.path.exists(sample_file):
            print(f"❌ Sample data file not found: {sample_file}")
            return False
        
        # Try to load the sample file
        df = pd.read_csv(sample_file)
        assert not df.empty, "Sample file should not be empty"
        assert 'date' in df.columns, "Sample file should have date column"
        assert 'description' in df.columns, "Sample file should have description column"
        assert 'amount' in df.columns, "Sample file should have amount column"
        
        print("✅ Sample data file tests passed")
        return True
    except Exception as e:
        print(f"❌ Sample data file test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Running Finance Analyzer Tests...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("DataLoader Tests", test_data_loader),
        ("DataCleaner Tests", test_data_cleaner),
        ("TransactionCategorizer Tests", test_categorizer),
        ("FinancialAnalyzer Tests", test_analyzer),
        ("ExpenseForecaster Tests", test_forecaster),
        ("Utils Tests", test_utils),
        ("Sample Data File Tests", test_sample_data_file)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\n🎯 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The Finance Analyzer is ready to use.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
