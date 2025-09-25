"""
Streamlit dashboard for the Finance Analyzer application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from cleaner import DataCleaner
from categorizer import TransactionCategorizer
from analyzer import FinancialAnalyzer
from forecaster import ExpenseForecaster


class FinanceDashboard:
    """Main dashboard class for the Finance Analyzer application."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.cleaner = DataCleaner()
        self.categorizer = TransactionCategorizer()
        self.analyzer = FinancialAnalyzer()
        self.forecaster = ExpenseForecaster()
        
        # Initialize session state
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'insights' not in st.session_state:
            st.session_state.insights = None
        if 'forecasts' not in st.session_state:
            st.session_state.forecasts = None
    
    def run(self):
        """Main function to run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Finance Analyzer",
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .insight-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #1f77b4;
            margin: 1rem 0;
            color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">💰 Personal Finance Analyzer</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        if st.session_state.processed_data is not None:
            self.show_dashboard()
        else:
            self.show_upload_section()
    
    def create_sidebar(self):
        """Create the sidebar with navigation and controls."""
        st.sidebar.title("📊 Navigation")
        
        # File upload section
        st.sidebar.header("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your bank statement",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file containing your bank transactions"
        )
        
        if uploaded_file is not None:
            self.process_uploaded_file(uploaded_file)
        
        # Sample data option
        if st.sidebar.button("📋 Use Sample Data"):
            self.load_sample_data()
        
        # Data info
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            st.sidebar.markdown("### 📈 Data Summary")
            st.sidebar.metric("Total Transactions", len(df))
            if 'date' in df.columns:
                date_range = df['date'].max() - df['date'].min()
                st.sidebar.metric("Date Range", f"{date_range.days} days")
        
        # Settings
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Settings")
        
        # Forecast settings
        forecast_days = st.sidebar.slider(
            "Forecast Period (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Number of days to forecast into the future"
        )
        
        if st.sidebar.button("🔮 Generate Forecast") and st.session_state.processed_data is not None:
            self.generate_forecast(forecast_days)
    
    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded file and clean the data."""
        try:
            with st.spinner("Processing your data..."):
                # Load data
                raw_data = self.data_loader.load_file(uploaded_file)
                
                if raw_data is None:
                    st.error("Failed to load the file. Please check the format.")
                    return
                
                # Validate columns
                validation = self.data_loader.validate_columns(raw_data)
                
                if not validation['is_valid']:
                    st.error("Required columns not found. Please ensure your file has date, description, and amount columns.")
                    st.write("Available columns:", validation['columns_found'])
                    return
                
                # Apply column mapping
                mapped_data = self.data_loader.apply_column_mapping(raw_data, validation['suggested_mapping'])
                
                # Clean data
                cleaned_data = self.cleaner.clean_data(mapped_data)
                cleaned_data = self.cleaner.add_derived_columns(cleaned_data)
                
                # Categorize transactions
                categorized_data = self.categorizer.categorize_transactions(cleaned_data)
                
                # Store in session state
                st.session_state.processed_data = categorized_data
                
                # Generate insights
                insights = self.analyzer.generate_insights(categorized_data)
                st.session_state.insights = insights
                
                st.success(f"✅ Successfully processed {len(categorized_data)} transactions!")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    def load_sample_data(self):
        """Load sample data for demonstration."""
        try:
            with st.spinner("Loading sample data..."):
                # Generate more comprehensive sample data
                sample_data = self.generate_sample_data()
                
                # Clean and process
                cleaned_data = self.cleaner.clean_data(sample_data)
                cleaned_data = self.cleaner.add_derived_columns(cleaned_data)
                categorized_data = self.categorizer.categorize_transactions(cleaned_data)
                
                # Store in session state
                st.session_state.processed_data = categorized_data
                
                # Generate insights
                insights = self.analyzer.generate_insights(categorized_data)
                st.session_state.insights = insights
                
                st.success("✅ Sample data loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
    
    def generate_sample_data(self):
        """Generate comprehensive sample transaction data."""
        np.random.seed(42)
        
        # Generate dates for last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        transactions = []
        
        for date in dates:
            # Random number of transactions per day (0-5)
            num_transactions = np.random.poisson(1.5)
            
            for _ in range(num_transactions):
                # Sample transaction types with realistic descriptions and amounts
                transaction_types = [
                    ('SWIGGY ORDER BANGALORE', -np.random.uniform(200, 800)),
                    ('UBER TRIP PAYMENT', -np.random.uniform(100, 500)),
                    ('AMAZON PURCHASE', -np.random.uniform(500, 3000)),
                    ('GROCERY STORE', -np.random.uniform(800, 2500)),
                    ('PETROL PUMP', -np.random.uniform(1000, 3000)),
                    ('NETFLIX SUBSCRIPTION', -199),
                    ('ELECTRICITY BILL', -np.random.uniform(1500, 4000)),
                    ('MOBILE RECHARGE', -np.random.uniform(200, 600)),
                    ('RESTAURANT PAYMENT', -np.random.uniform(800, 2500)),
                    ('MEDICAL STORE', -np.random.uniform(300, 1500)),
                    ('SALARY CREDIT', np.random.uniform(50000, 80000)),
                    ('ATM WITHDRAWAL', -np.random.uniform(2000, 10000)),
                    ('RENT PAYMENT', -15000),
                ]
                
                # Weight probabilities (salary less frequent)
                weights = [0.15, 0.12, 0.15, 0.1, 0.08, 0.02, 0.03, 0.03, 0.08, 0.04, 0.02, 0.05, 0.13]
                
                # Select random transaction
                selected_idx = np.random.choice(
                    len(transaction_types),
                    p=weights/np.sum(weights)
                )
                desc, base_amount = transaction_types[selected_idx]
                
                # Add some randomness to amount (except for fixed amounts like Netflix, Rent)
                if base_amount == -199 or base_amount == -15000:  # Fixed amounts
                    amount = base_amount
                else:
                    amount = base_amount * np.random.uniform(0.8, 1.2)
                
                transactions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'description': desc,
                    'amount': round(amount, 2)
                })
        
        return pd.DataFrame(transactions)
    
    def show_upload_section(self):
        """Show the initial upload section when no data is loaded."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h2>🚀 Get Started</h2>
                <p>Upload your bank statement or use sample data to begin analyzing your finances.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 📋 Supported File Formats
            - **CSV files** (.csv)
            - **Excel files** (.xlsx, .xls)
            
            ### 📊 Required Columns
            Your file should contain:
            - **Date** (transaction date)
            - **Description** (transaction description)
            - **Amount** (transaction amount - positive for income, negative for expenses)
            """)
    
    def show_dashboard(self):
        """Show the main dashboard with all visualizations and insights."""
        df = st.session_state.processed_data
        insights = st.session_state.insights
        
        # Overview metrics
        self.show_overview_metrics(insights)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "💰 Spending Analysis", "📈 Trends", "🔮 Forecasting", "📋 Transactions"
        ])
        
        with tab1:
            self.show_overview_tab(df, insights)
        
        with tab2:
            self.show_spending_analysis_tab(df, insights)
        
        with tab3:
            self.show_trends_tab(df, insights)
        
        with tab4:
            self.show_forecasting_tab(df)
        
        with tab5:
            self.show_transactions_tab(df)
    
    def show_overview_metrics(self, insights):
        """Show key financial metrics at the top of the dashboard."""
        overview = insights.get('overview', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_income = overview.get('total_income', 0)
            st.metric(
                label="💰 Total Income",
                value=f"₹{total_income:,.0f}",
                delta=None
            )
        
        with col2:
            total_expenses = overview.get('total_expenses', 0)
            st.metric(
                label="💸 Total Expenses",
                value=f"₹{total_expenses:,.0f}",
                delta=None
            )
        
        with col3:
            net_flow = overview.get('net_cash_flow', 0)
            delta_color = "normal" if net_flow >= 0 else "inverse"
            st.metric(
                label="📊 Net Cash Flow",
                value=f"₹{net_flow:,.0f}",
                delta=f"{'Surplus' if net_flow >= 0 else 'Deficit'}",
                delta_color=delta_color
            )
        
        with col4:
            savings_rate = insights.get('savings_rate', {}).get('overall_savings_rate', 0)
            st.metric(
                label="💡 Savings Rate",
                value=f"{savings_rate:.1f}%",
                delta="Target: 20%"
            )
    
    def show_overview_tab(self, df, insights):
        """Show overview tab with key insights and summary."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Monthly trends chart
            monthly_trends = insights.get('monthly_trends')
            if not monthly_trends.empty:
                fig = self.create_monthly_trends_chart(monthly_trends)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key insights
            st.markdown("### 🔍 Key Insights")
            insights_summary = self.analyzer.generate_insights_summary(insights)
            st.markdown(f'<div class="insight-box">{insights_summary}</div>', unsafe_allow_html=True)
        
        # Spending by category pie chart
        spending_by_category = insights.get('spending_by_category')
        if not spending_by_category.empty:
            fig = self.create_category_pie_chart(spending_by_category)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_spending_analysis_tab(self, df, insights):
        """Show detailed spending analysis."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Category breakdown
            spending_by_category = insights.get('spending_by_category')
            if not spending_by_category.empty:
                st.subheader("💰 Spending by Category")
                
                # Bar chart
                fig = px.bar(
                    spending_by_category.head(10),
                    x='total_spent',
                    y='category',
                    orientation='h',
                    title="Top 10 Spending Categories",
                    labels={'total_spent': 'Amount Spent (₹)', 'category': 'Category'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(
                    spending_by_category[['category', 'total_spent', 'transaction_count', 'percentage']],
                    use_container_width=True
                )
        
        with col2:
            # Weekly patterns
            weekly_patterns = insights.get('weekly_patterns')
            if not weekly_patterns.empty:
                st.subheader("📅 Weekly Spending Patterns")
                
                fig = px.bar(
                    weekly_patterns,
                    x='weekday',
                    y='total_spent',
                    title="Spending by Day of Week",
                    labels={'total_spent': 'Amount Spent (₹)', 'weekday': 'Day of Week'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Top transactions
        top_transactions = insights.get('top_transactions', {})
        if top_transactions:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💸 Top Expenses")
                top_expenses = top_transactions.get('top_expenses')
                if not top_expenses.empty:
                    st.dataframe(top_expenses, use_container_width=True)
            
            with col2:
                st.subheader("💰 Top Income")
                top_income = top_transactions.get('top_income')
                if not top_income.empty:
                    st.dataframe(top_income, use_container_width=True)
    
    def show_trends_tab(self, df, insights):
        """Show trends and patterns analysis."""
        # Monthly trends
        monthly_trends = insights.get('monthly_trends')
        if not monthly_trends.empty:
            st.subheader("📈 Monthly Financial Trends")
            
            # Create comprehensive monthly chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Income vs Expenses', 'Savings Trend'),
                vertical_spacing=0.1
            )
            
            # Income vs Expenses
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends['year_month'].astype(str),
                    y=monthly_trends['income'],
                    mode='lines+markers',
                    name='Income',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends['year_month'].astype(str),
                    y=monthly_trends['expenses'],
                    mode='lines+markers',
                    name='Expenses',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Savings trend
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends['year_month'].astype(str),
                    y=monthly_trends['savings'],
                    mode='lines+markers',
                    name='Savings',
                    line=dict(color='blue'),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Financial Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Category trends over time
        if 'category' in df.columns and 'date' in df.columns:
            st.subheader("📊 Category Trends Over Time")
            
            # Get monthly category data
            category_trends = self.categorizer.get_monthly_category_trends(df)
            if not category_trends.empty:
                # Select top categories for visualization
                top_categories = insights.get('spending_by_category', pd.DataFrame())
                if not top_categories.empty:
                    top_5_categories = top_categories.head(5)['category'].tolist()
                    
                    fig = go.Figure()
                    
                    for category in top_5_categories:
                        if category in category_trends.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=category_trends['year_month'].astype(str),
                                    y=category_trends[category],
                                    mode='lines+markers',
                                    name=category
                                )
                            )
                    
                    fig.update_layout(
                        title="Top 5 Categories - Monthly Spending Trends",
                        xaxis_title="Month",
                        yaxis_title="Amount Spent (₹)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_forecasting_tab(self, df):
        """Show forecasting results and predictions."""
        st.subheader("🔮 Expense Forecasting")
        
        if st.session_state.forecasts is None:
            st.info("Click 'Generate Forecast' in the sidebar to see predictions for future expenses.")
            return
        
        forecasts = st.session_state.forecasts
        
        if 'error' in forecasts:
            st.error(f"Forecasting error: {forecasts['error']}")
            return
        
        # Display forecast results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Forecast visualization
            self.show_forecast_chart(df, forecasts)
        
        with col2:
            # Forecast summary
            self.show_forecast_summary(forecasts)
        
        # Model comparison
        if len(forecasts) > 1:
            st.subheader("🔬 Model Comparison")
            self.show_model_comparison(forecasts)
    
    def show_transactions_tab(self, df):
        """Show detailed transaction data with filtering options."""
        st.subheader("📋 Transaction Details")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date filter
            if 'date' in df.columns:
                date_range = st.date_input(
                    "Date Range",
                    value=(df['date'].min(), df['date'].max()),
                    min_value=df['date'].min(),
                    max_value=df['date'].max()
                )
        
        with col2:
            # Category filter
            if 'category' in df.columns:
                categories = ['All'] + sorted(df['category'].unique().tolist())
                selected_category = st.selectbox("Category", categories)
        
        with col3:
            # Transaction type filter
            transaction_types = ['All', 'Income', 'Expenses']
            selected_type = st.selectbox("Transaction Type", transaction_types)
        
        # Apply filters
        filtered_df = df.copy()
        
        if 'date' in df.columns and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= date_range[0]) &
                (filtered_df['date'].dt.date <= date_range[1])
            ]
        
        if 'category' in df.columns and selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if selected_type == 'Income':
            filtered_df = filtered_df[filtered_df['amount'] > 0]
        elif selected_type == 'Expenses':
            filtered_df = filtered_df[filtered_df['amount'] < 0]
        
        # Display filtered data
        st.write(f"Showing {len(filtered_df)} transactions")
        
        # Format the dataframe for display
        display_df = filtered_df.copy()
        if 'amount' in display_df.columns:
            display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.2f}")
        
        st.dataframe(
            display_df[['date', 'description', 'amount', 'category']],
            use_container_width=True
        )
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def generate_forecast(self, days):
        """Generate expense forecasts."""
        try:
            with st.spinner(f"Generating {days}-day forecast..."):
                df = st.session_state.processed_data
                forecasts = self.forecaster.forecast_next_period(df, days=days)
                st.session_state.forecasts = forecasts
                st.success("✅ Forecast generated successfully!")
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
    
    def create_monthly_trends_chart(self, monthly_trends):
        """Create monthly trends chart."""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_trends['year_month'].astype(str),
                y=monthly_trends['income'],
                mode='lines+markers',
                name='Income',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_trends['year_month'].astype(str),
                y=monthly_trends['expenses'],
                mode='lines+markers',
                name='Expenses',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            )
        )
        
        fig.update_layout(
            title="Monthly Income vs Expenses",
            xaxis_title="Month",
            yaxis_title="Amount (₹)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_category_pie_chart(self, spending_by_category):
        """Create category spending pie chart."""
        # Take top 8 categories and group rest as "Others"
        top_categories = spending_by_category.head(8)
        others_amount = spending_by_category.iloc[8:]['total_spent'].sum() if len(spending_by_category) > 8 else 0
        
        if others_amount > 0:
            others_row = pd.DataFrame({
                'category': ['Others'],
                'total_spent': [others_amount],
                'percentage': [others_amount / spending_by_category['total_spent'].sum() * 100]
            })
            plot_data = pd.concat([top_categories, others_row], ignore_index=True)
        else:
            plot_data = top_categories
        
        fig = px.pie(
            plot_data,
            values='total_spent',
            names='category',
            title="Spending Distribution by Category",
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig
    
    def show_forecast_chart(self, df, forecasts):
        """Show forecast visualization chart."""
        fig = go.Figure()
        
        # Historical data
        if 'date' in df.columns and 'amount' in df.columns:
            historical_expenses = df[df['amount'] < 0].copy()
            historical_expenses['abs_amount'] = historical_expenses['amount'].abs()
            daily_historical = historical_expenses.groupby('date')['abs_amount'].sum().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_historical['date'],
                    y=daily_historical['abs_amount'],
                    mode='lines',
                    name='Historical Expenses',
                    line=dict(color='blue', width=2)
                )
            )
        
        # Forecast data
        colors = ['red', 'green', 'orange', 'purple']
        color_idx = 0
        
        for model_name, result in forecasts.items():
            if 'forecast' in result and not result['forecast'].empty:
                forecast_df = result['forecast']
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['predicted_amount'],
                        mode='lines+markers',
                        name=f'{model_name.replace("_", " ").title()} Forecast',
                        line=dict(color=colors[color_idx % len(colors)], width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )
                color_idx += 1
        
        fig.update_layout(
            title="Expense Forecast vs Historical Data",
            xaxis_title="Date",
            yaxis_title="Daily Expenses (₹)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_forecast_summary(self, forecasts):
        """Show forecast summary metrics."""
        st.markdown("### 📊 Forecast Summary")
        
        summary = self.forecaster.get_monthly_forecast_summary(forecasts)
        
        for model_name, metrics in summary.items():
            with st.expander(f"{model_name.replace('_', ' ').title()} Model"):
                st.metric("Total Predicted", f"₹{metrics['total_predicted']:,.0f}")
                st.metric("Daily Average", f"₹{metrics['daily_average']:,.0f}")
                
                if 'monthly_breakdown' in metrics:
                    st.write("**Monthly Breakdown:**")
                    for month, amount in metrics['monthly_breakdown'].items():
                        st.write(f"• {month}: ₹{amount:,.0f}")
    
    def show_model_comparison(self, forecasts):
        """Show comparison between different forecasting models."""
        comparison_data = []
        
        for model_name, result in forecasts.items():
            if 'training_metrics' in result:
                metrics = result['training_metrics']
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'MAE': metrics.get('mae', 'N/A'),
                    'RMSE': metrics.get('rmse', 'N/A'),
                    'Training Samples': metrics.get('training_samples', 'N/A')
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)


def main():
    """Main function to run the dashboard."""
    dashboard = FinanceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
