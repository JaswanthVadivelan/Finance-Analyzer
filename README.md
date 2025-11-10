# 💰 Personal Finance Analyzer

A comprehensive Python-based finance analyzer that helps you understand your spending patterns, categorize transactions, and forecast future expenses using machine learning.

## 🚀 Features

### Core Features
- ✅ **File Upload Support**: CSV and Excel bank statements
- ✅ **Smart Data Cleaning**: Handles various date formats, currency symbols, and data inconsistencies
- ✅ **Automatic Categorization**: AI-powered transaction categorization (Food, Transport, Shopping, etc.)
- ✅ **Interactive Dashboard**: Beautiful Streamlit-based web interface
- ✅ **Financial Insights**: Monthly trends, spending patterns, and savings analysis
- ✅ **Expense Forecasting**: ML-based predictions using Linear Regression, Random Forest, and ARIMA

### Advanced Features
- 📊 **Visual Analytics**: Interactive charts and graphs using Plotly
- 🔍 **Anomaly Detection**: Identify unusual transactions
- 📈 **Trend Analysis**: Weekly and monthly spending patterns
- 💡 **Savings Rate Calculation**: Track your financial health
- 📋 **Transaction Filtering**: Advanced search and filter capabilities
- 📥 **Data Export**: Download filtered transaction data

## 🏗️ Project Structure

```
finance-analyzer/
│
├── data/                          # Sample data and user uploads
│   ├── sample_bank_statement.csv  # Sample data for testing
│   └── .gitkeep
│
├── notebooks/                     # Jupyter notebooks for analysis
│   └── .gitkeep
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # CSV/Excel file handling
│   ├── cleaner.py               # Data cleaning and preprocessing
│   ├── categorizer.py           # Transaction categorization
│   ├── analyzer.py              # Financial analysis and insights
│   ├── forecaster.py            # ML-based expense forecasting
│   ├── dashboard.py             # Streamlit dashboard
│   └── utils.py                 # Utility functions
│
├── app.py                       # Main application entry point
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free!)

1. **Fork this repository** to your GitHub account
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Click "New app"**
4. **Connect your GitHub repository**
5. **Set the main file path**: `app.py`
6. **Click "Deploy"**

Your app will be live at: `https://[your-username]-finance-analyzer-[random-string].streamlit.app`

## 📊 How to Use

### 1. Upload Your Data
- **Option 1**: Upload your bank statement (CSV or Excel format) You can also use our provided sample CSV file to test the application.
- **Option 2**: Use the sample data to explore features

### 2. Required Data Format
Your file should contain these columns:
- **Date**: Transaction date (various formats supported)
- **Description**: Transaction description
- **Amount**: Transaction amount (positive for income, negative for expenses)

Example:
```csv
date,description,amount
2024-01-15,SWIGGY BANGALORE,-450.50
2024-01-15,SALARY CREDIT,75000.00
2024-01-16,UBER TRIP PAYMENT,-280.00
```

### 3. Explore Your Finances
- **Overview Tab**: Key metrics and monthly trends
- **Spending Analysis**: Category breakdown and patterns
- **Trends**: Historical analysis and patterns
- **Forecasting**: Future expense predictions
- **Transactions**: Detailed transaction view with filters

## 🎯 Key Insights You'll Get

### Financial Overview
- Total income vs expenses
- Net cash flow
- Savings rate analysis
- Transaction count and averages

### Spending Analysis
- **Category Breakdown**: See where your money goes
- **Top Expenses**: Identify your biggest transactions
- **Weekly Patterns**: Understand when you spend most
- **Monthly Trends**: Track spending over time

### Smart Categorization
Transactions are automatically categorized into:
- 🍔 Food & Dining
- 🚗 Transportation
- 🛍️ Shopping
- 💡 Bills & Utilities
- 🎬 Entertainment
- 🏥 Healthcare
- 📚 Education
- 💰 Investment & Savings
- 🏠 Rent & Housing
- 💄 Personal Care
- 💵 Income
- 🏧 ATM & Banking

### Forecasting Models
- **Linear Regression**: Trend-based predictions
- **Random Forest**: Advanced ML predictions
- **Ensemble**: Combined model predictions

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning models
- **statsmodels**: Statistical analysis and ARIMA
- **openpyxl**: Excel file support

### Data Processing Pipeline
1. **Data Loading**: Support for CSV/Excel with encoding detection
2. **Data Validation**: Column mapping and validation
3. **Data Cleaning**: Date parsing, amount cleaning, duplicate removal
4. **Categorization**: Regex and keyword-based classification
5. **Analysis**: Statistical analysis and insight generation
6. **Forecasting**: ML model training and prediction

## 📈 Sample Analysis Results

With the sample data, you can expect to see:
- **Monthly spending trends** over the past 6 months
- **Category distribution** showing top spending areas
- **Savings rate** calculation and recommendations
- **Weekly patterns** identifying high-spending days
- **Expense forecasts** for the next 30 days

## 🚀 Advanced Usage

### Custom Categories
You can add custom transaction categories by modifying the `categorizer.py` file:

```python
categorizer.add_custom_category(
    category_name="Custom Category",
    keywords=["keyword1", "keyword2"],
    patterns=[r".*pattern.*"]
)
```

### Export Data
- Use the download button in the Transactions tab
- Export filtered transaction data as CSV
- Save insights and analysis results

### Forecasting Settings
- Adjust forecast period (7-90 days)
- Compare different ML models
- View model performance metrics

## 🎨 Dashboard Features

### Interactive Charts
- **Pie Charts**: Category distribution
- **Line Charts**: Monthly trends
- **Bar Charts**: Weekly patterns and top categories
- **Forecast Charts**: Historical vs predicted expenses

### Responsive Design
- Works on desktop and mobile devices
- Clean, modern interface
- Intuitive navigation

## 🔍 Troubleshooting

### Common Issues

**File Upload Errors**
- Ensure your file has the required columns (date, description, amount)
- Check file format (CSV or Excel)
- Verify data encoding (UTF-8 recommended)

**Date Parsing Issues**
- The system supports multiple date formats
- Ensure dates are in a recognizable format
- Check for missing or invalid dates

**Amount Parsing Issues**
- Remove currency symbols if causing issues
- Ensure amounts are numeric
- Use negative values for expenses, positive for income

**Performance Issues**
- Large files (>10MB) may take longer to process
- Consider filtering data to recent months for better performance

### Getting Help
If you encounter issues:
1. Check the error messages in the app
2. Verify your data format matches the requirements
3. Try using the sample data to test functionality

## 🔮 Future Enhancements

### Planned Features
- **Budget Management**: Set and track budgets by category
- **Goal Setting**: Financial goal tracking
- **Multi-Account Support**: Analyze multiple bank accounts
- **PDF Statement Parsing**: Direct PDF bank statement upload
- **Mobile App**: React Native mobile application
- **API Integration**: Connect directly to bank APIs
- **Advanced ML**: Deep learning models for better predictions

### Contributing
This project is designed for learning and portfolio purposes. Feel free to:
- Add new features
- Improve existing functionality
- Enhance the UI/UX
- Add more visualization options

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Plotly**: For beautiful interactive visualizations
- **Pandas**: For powerful data manipulation capabilities
- **Scikit-learn**: For machine learning functionality

---

## 🎯 Perfect for Your Portfolio

This project demonstrates:
- **Full-Stack Development**: Python backend with web frontend
- **Data Science Skills**: Data cleaning, analysis, and ML
- **Financial Domain Knowledge**: Understanding of personal finance
- **Modern Tech Stack**: Streamlit, Plotly, Pandas, Scikit-learn
- **Professional Code Structure**: Modular, documented, and maintainable
- **Real-World Application**: Solves actual personal finance problems

### Resume Highlights
- Built end-to-end finance analytics platform
- Implemented ML-based expense forecasting
- Created interactive data visualizations
- Designed automated transaction categorization system
- Developed responsive web dashboard

**Start analyzing your finances today! 💰📊**
