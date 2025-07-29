import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.api import Holt # <-- ADD THIS IMPORT
from flask.json.provider import DefaultJSONProvider

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert NaN and inf to None
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.json_encoder = NumpyEncoder

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_str(value):
    """Safely convert any value to string"""
    if value is None:
        return "None"
    return str(value)

def identify_column_types(df):
    """Identify different types of columns in the dataset"""
    column_types = {
        'revenue_columns': [],
        'date_columns': [],
        'location_columns': [],
        'category_columns': [],
        'financial_columns': [],
        'all_columns': list(df.columns)
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Revenue columns
        if any(keyword in col_lower for keyword in ['revenue', 'amount', 'value', 'price', 'cost', 'payment']):
            column_types['revenue_columns'].append(col)
        
        # Financial columns (including COD, Customer Value, Extra amounts)
        if any(keyword in col_lower for keyword in ['cod', 'customer value', 'extra amount', 'extra bill', 'tx value', 'bill']):
            column_types['financial_columns'].append(col)
        
        # Date columns
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated']):
            column_types['date_columns'].append(col)
        
        # Location columns
        if any(keyword in col_lower for keyword in ['pincode', 'zipcode', 'city', 'state', 'location', 'address', 'pin', 'area', 'region', 'district']):
            column_types['location_columns'].append(col)
        
        # Category columns
        if any(keyword in col_lower for keyword in ['category', 'type', 'status', 'order type', 'order category']):
            column_types['category_columns'].append(col)
    
    # Exclude location columns from revenue and financial columns
    location_set = set(column_types['location_columns'])
    column_types['revenue_columns'] = [col for col in column_types['revenue_columns'] if col not in location_set]
    column_types['financial_columns'] = [col for col in column_types['financial_columns'] if col not in location_set]

    return column_types

def generate_basic_analytics(df):
    """Generate basic dataset analytics"""
    try:
        analytics = {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'column_info': {}
        }
        
        for col in df.columns:
            col_data = df[col]
            analytics['column_info'][str(col)] = {
                'data_type': str(col_data.dtype),
                'unique_values': int(col_data.nunique()),
                'missing_values': int(col_data.isnull().sum()),
                'sample_values': [str(x) for x in col_data.dropna().head(5).tolist()] if col_data.nunique() <= 10 else []
            }
        
        return analytics
    except Exception as e:
        logger.error("Error in basic analytics: %s", str(e))
        return {'error': str(e)}

def generate_revenue_insights(df, column_types):
    """Generate comprehensive revenue and financial insights"""
    try:
        insights = {}
        
        revenue_cols = column_types.get('revenue_columns', [])
        financial_cols = column_types.get('financial_columns', [])
        date_cols = column_types.get('date_columns', [])
        location_cols = column_types.get('location_columns', [])
        category_cols = column_types.get('category_columns', [])
        
        all_financial_cols = revenue_cols + financial_cols
        
        if not all_financial_cols:
            return {'message': 'No financial columns found'}
        
        insights['financial_summary'] = {}
        
        # Analyze each financial column
        for col in all_financial_cols:
            try:
                # Convert to numeric, handling non-numeric values
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                numeric_data = numeric_data.dropna()
                
                if len(numeric_data) > 0:
                    insights['financial_summary'][str(col)] = {
                        'total': float(numeric_data.sum()),
                        'average': float(numeric_data.mean()),
                        'median': float(numeric_data.median()),
                        'min': float(numeric_data.min()),
                        'max': float(numeric_data.max()),
                        'count': int(len(numeric_data)),
                        'std_dev': float(numeric_data.std())
                    }
            except Exception as e:
                logger.error("Error processing column %s: %s", safe_str(col), str(e))
                continue
        
        # Calculate grand total revenue across all financial columns
        grand_total = sum(
            v['total'] for v in insights['financial_summary'].values() if 'total' in v
        )
        insights['grand_total_revenue'] = grand_total

        # Time-based analysis if date columns exist
        if date_cols:
            insights['time_analysis'] = {}
            for date_col in date_cols[:1]:  # Use first date column
                try:
                    df_copy = df.copy()
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    df_copy = df_copy.dropna(subset=[date_col])
                    
                    if len(df_copy) > 0:
                        # Get only numeric financial columns for aggregation
                        numeric_financial_cols = []
                        for col in all_financial_cols:
                            if col in df_copy.columns:
                                try:
                                    # Try to convert to numeric to see if it's summable
                                    pd.to_numeric(df_copy[col], errors='coerce')
                                    numeric_financial_cols.append(col)
                                except Exception as e:
                                    logger.error("Error converting column %s to numeric: %s", safe_str(col), str(e))
                                    continue
                        
                        if numeric_financial_cols:
                            # Daily trends
                            daily_data = df_copy.groupby(df_copy[date_col].dt.date).agg({
                                col: 'sum' for col in numeric_financial_cols
                            }).reset_index()
                        else:
                            # If no numeric columns, just get the count
                            daily_data = df_copy.groupby(df_copy[date_col].dt.date).size().reset_index(name='order_count')
                        
                        # Convert to JSON-safe format
                        daily_records = []
                        for _, row in daily_data.iterrows():
                            record = {}
                            for col in row.index:
                                if pd.isna(row[col]):
                                    record[str(col)] = None
                                elif isinstance(row[col], (np.integer, np.floating)):
                                    record[str(col)] = float(row[col])
                                else:
                                    record[str(col)] = str(row[col])
                            daily_records.append(record)
                        
                        insights['time_analysis']['daily_trends'] = daily_records
            
            # Monthly trends
                        if numeric_financial_cols:
                            monthly_data = df_copy.groupby(df_copy[date_col].dt.to_period('M')).agg({
                                col: 'sum' for col in numeric_financial_cols
                            }).reset_index()
                        else:
                            # If no numeric columns, just get the count
                            monthly_data = df_copy.groupby(df_copy[date_col].dt.to_period('M')).size().reset_index(name='order_count')
                        
                        # Convert to JSON-safe format
                        monthly_records = []
                        for _, row in monthly_data.iterrows():
                            record = {}
                            for col in row.index:
                                if pd.isna(row[col]):
                                    record[str(col)] = None
                                elif isinstance(row[col], (np.integer, np.floating)):
                                    record[str(col)] = float(row[col])
                                else:
                                    record[str(col)] = str(row[col])
                            monthly_records.append(record)
                        
                        insights['time_analysis']['monthly_trends'] = monthly_records
                        
                except Exception as e:
                    logger.error("Error in time analysis for column %s: %s", safe_str(date_col), str(e))
                    continue
        
        # Location-based analysis
        if location_cols:
            insights['location_analysis'] = {}
            for loc_col in location_cols[:1]:  # Use first location column
                try:
                    # Check if location column has data
                    if df[loc_col].isnull().all():
                        insights['location_analysis']['message'] = f'Location column "{safe_str(loc_col)}" has no data'
                        continue
                    
                    # Get unique locations
                    unique_locations = df[loc_col].dropna().unique()
                    if len(unique_locations) == 0:
                        insights['location_analysis']['message'] = f'No valid location data found in column "{safe_str(loc_col)}"'
                        continue
                    
                    # Group by location and aggregate financial data
                    numeric_financial_cols = []
                    for col in all_financial_cols:
                        if col in df.columns:
                            try:
                                # Try to convert to numeric to see if it's summable
                                pd.to_numeric(df[col], errors='coerce')
                                numeric_financial_cols.append(col)
                            except Exception as e:
                                logger.error("Error converting column %s to numeric: %s", safe_str(col), str(e))
                                continue
                    
                    if numeric_financial_cols:
                        location_data = df.groupby(loc_col).agg({
                            col: 'sum' for col in numeric_financial_cols
                        }).reset_index()
                    else:
                        # If no numeric columns, just get the count
                        location_data = df.groupby(loc_col).size().reset_index(name='order_count')
                    
                    # Add count of orders per location (only if we have financial data)
                    if numeric_financial_cols:
                        try:
                            order_counts = df.groupby(loc_col).size().reset_index(name='order_count')
                            location_data = location_data.merge(order_counts, on=loc_col, how='left')
                        except Exception as e:
                            logger.error("Error merging order counts: %s", str(e))
                            # Continue without order counts
                    
                    # Convert to JSON-safe format
                    location_records = []
                    try:
                        for _, row in location_data.iterrows():
                            record = {}
                            for col in row.index:
                                try:
                                    if pd.isna(row[col]):
                                        record[str(col)] = None
                                    elif isinstance(row[col], (np.integer, np.floating)):
                                        record[str(col)] = float(row[col])
                                    else:
                                        record[str(col)] = str(row[col])
                                except Exception as e:
                                    logger.error("Error converting column %s value: %s", safe_str(col), str(e))
                                    record[str(col)] = str(row[col]) if row[col] is not None else None
                            location_records.append(record)
                    except Exception as e:
                        logger.error("Error converting location records: %s", str(e))
                        location_records = []
                    
                    insights['location_analysis']['by_location'] = location_records
                    insights['location_analysis']['location_column'] = loc_col
                    insights['location_analysis']['total_locations'] = len(unique_locations)
                    
                except Exception as e:
                    logger.error("Error in location analysis for column %s: %s", safe_str(loc_col), str(e))
                    insights['location_analysis']['error'] = str(e)
                    continue
        else:
            insights['location_analysis'] = {'message': 'No location columns detected in the dataset'}
        
        # Category-based analysis
        if category_cols:
            insights['category_analysis'] = {}
            for cat_col in category_cols[:1]:  # Use first category column
                try:
                    # Get only numeric financial columns for aggregation
                    numeric_financial_cols = []
                    for col in all_financial_cols:
                        if col in df.columns:
                            try:
                                # Try to convert to numeric to see if it's summable
                                pd.to_numeric(df[col], errors='coerce')
                                numeric_financial_cols.append(col)
                            except Exception as e:
                                logger.error("Error converting column %s to numeric: %s", safe_str(col), str(e))
                                continue
                    
                    if numeric_financial_cols:
                        category_data = df.groupby(cat_col).agg({
                            col: 'sum' for col in numeric_financial_cols
                        }).reset_index()
                    else:
                        # If no numeric columns, just get the count
                        category_data = df.groupby(cat_col).size().reset_index(name='order_count')
                    
                    # Convert to JSON-safe format
                    category_records = []
                    for _, row in category_data.iterrows():
                        record = {}
                        for col in row.index:
                            if pd.isna(row[col]):
                                record[str(col)] = None
                            elif isinstance(row[col], (np.integer, np.floating)):
                                record[str(col)] = float(row[col])
                            else:
                                record[str(col)] = str(row[col])
                        category_records.append(record)
                    
                    insights['category_analysis']['by_category'] = category_records
                    
                except Exception as e:
                    logger.error("Error in category analysis for column %s: %s", safe_str(cat_col), str(e))
                    continue
        
        return insights
                
    except Exception as e:
        logger.error("Error in revenue insights: %s", str(e))
        return {'error': str(e)}

def generate_column_insights(df, column_name):
    """Generate detailed insights for a specific column"""
    try:
        col_data = df[column_name]
        insights = {
            'column_name': str(column_name),
            'data_type': str(col_data.dtype),
            'total_values': int(len(col_data)),
            'unique_values': int(col_data.nunique()),
            'missing_values': int(col_data.isnull().sum()),
            'missing_percentage': round((col_data.isnull().sum() / len(col_data)) * 100, 2)
        }
        
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            numeric_data = col_data.dropna()
            if len(numeric_data) > 0:
                insights.update({
                    'min_value': float(numeric_data.min()),
                    'max_value': float(numeric_data.max()),
                    'average': float(numeric_data.mean()),
                    'median': float(numeric_data.median()),
                    'std_deviation': float(numeric_data.std()),
                    'quartiles': {
                        'q1': float(numeric_data.quantile(0.25)),
                        'q2': float(numeric_data.quantile(0.50)),
                        'q3': float(numeric_data.quantile(0.75))
                    }
                })
        
        # Handle categorical/text columns
        else:
            value_counts = col_data.value_counts().head(10)
            insights['top_values'] = value_counts.to_dict()
            insights['unique_count'] = col_data.nunique()
        
        # Sample data
        sample_data = [str(x) for x in col_data.dropna().head(10).tolist()]
        insights['sample_data'] = sample_data
        
        return insights
        
    except Exception as e:
        logger.error("Error in column insights: %s", str(e))
        return {'error': str(e)}

def generate_advanced_analytics(df, column_types):
    """Generate advanced analytics including trends, performance, and risk assessment"""
    try:
        insights = {}
        
        # Performance Analytics
        insights['performance_metrics'] = generate_performance_metrics(df, column_types)
        
        # Trend Analysis
        insights['trend_analysis'] = generate_trend_analysis(df, column_types)
        
        # Risk Assessment
        insights['risk_assessment'] = generate_risk_assessment(df, column_types)
        
        # Customer Analytics
        insights['customer_analytics'] = generate_customer_analytics(df, column_types)
        
        # Operational Insights
        insights['operational_insights'] = generate_operational_insights(df, column_types)
        
        # Financial Deep Dive
        insights['financial_deep_dive'] = generate_financial_deep_dive(df, column_types)
        
        return insights
    except Exception as e:
        logger.error("Error in advanced analytics: %s", str(e))
        return {'error': str(e)}

def generate_performance_metrics(df, column_types):
    """Generate comprehensive performance metrics and KPIs with optimized performance"""
    try:
        metrics = {}
        
        # Pre-calculate common aggregations for efficiency
        customer_counts = df.get('Customer Name', pd.Series()).value_counts() if 'Customer Name' in df.columns else pd.Series()
        technician_counts = df.get('Technician Name', pd.Series()).value_counts() if 'Technician Name' in df.columns else pd.Series()
        facility_counts = df.get('Facility Name', pd.Series()).value_counts() if 'Facility Name' in df.columns else pd.Series()
        service_counts = df.get('Service Type', pd.Series()).value_counts() if 'Service Type' in df.columns else pd.Series()
        
        # 1. Basic Performance Metrics (Optimized)
        metrics['total_orders'] = int(len(df))
        metrics['unique_customers'] = int(len(customer_counts))
        metrics['unique_technicians'] = int(len(technician_counts))
        metrics['unique_facilities'] = int(len(facility_counts))
        metrics['unique_service_types'] = int(len(service_counts))
        
        # 2. Financial Performance Analysis
        financial_cols = column_types.get('financial_columns', [])
        financial_metrics = {}
        
        if financial_cols:
            total_revenue = 0
            financial_data = {}
            
            for col in financial_cols:
                if col in df.columns:
                    try:
                        numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            col_total = numeric_data.sum()
                            col_avg = numeric_data.mean()
                            col_median = numeric_data.median()
                            col_max = numeric_data.max()
                            col_min = numeric_data.min()
                            
                            financial_data[col] = {
                                'total': float(col_total),
                                'average': float(col_avg),
                                'median': float(col_median),
                                'max': float(col_max),
                                'min': float(col_min),
                                'count': int(len(numeric_data))
                            }
                            total_revenue += col_total
                    except Exception as e:
                        logger.error("Error processing financial column %s: %s", safe_str(col), str(e))
                        continue
            
            financial_metrics['total_revenue'] = float(total_revenue)
            financial_metrics['average_order_value'] = float(total_revenue / len(df)) if len(df) > 0 else 0
            financial_metrics['detailed_analysis'] = financial_data
            
            # Revenue per customer
            if 'Customer Name' in df.columns and total_revenue > 0:
                try:
                    customer_revenue = df.groupby('Customer Name').agg({
                        col: 'sum' for col in financial_cols if col in df.columns
                    }).sum(axis=1)
                    financial_metrics['average_revenue_per_customer'] = float(customer_revenue.mean()) if len(customer_revenue) > 0 else 0
                    financial_metrics['top_customer_revenue'] = float(customer_revenue.max()) if len(customer_revenue) > 0 else 0
                except Exception as e:
                    logger.error("Error calculating customer revenue: %s", str(e))
                    financial_metrics['average_revenue_per_customer'] = 0
                    financial_metrics['top_customer_revenue'] = 0
        
        metrics['financial_metrics'] = financial_metrics
        
        # 3. Operational Performance Metrics
        operational_metrics = {}
        
        # Completion Rate
        if 'Order Status' in df.columns:
            completed_orders = df[df['Order Status'].str.contains('completed|done|finished', case=False, na=False)]
            cancelled_orders = df[df['Order Status'].str.contains('cancelled|cancel', case=False, na=False)]
            pending_orders = df[~df['Order Status'].str.contains('completed|done|finished|cancelled|cancel', case=False, na=False)]
            
            operational_metrics['completion_rate'] = round((len(completed_orders) / len(df)) * 100, 2) if len(df) > 0 else 0
            operational_metrics['cancellation_rate'] = round((len(cancelled_orders) / len(df)) * 100, 2) if len(df) > 0 else 0
            operational_metrics['pending_rate'] = round((len(pending_orders) / len(df)) * 100, 2) if len(df) > 0 else 0
            operational_metrics['completed_orders'] = int(len(completed_orders))
            operational_metrics['cancelled_orders'] = int(len(cancelled_orders))
            operational_metrics['pending_orders'] = int(len(pending_orders))
        
        # Processing Time Analysis
        if 'Created At' in df.columns and 'Completed Date' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                
                if len(df_copy) > 0:
                    df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                    
                    operational_metrics['avg_processing_time_hours'] = round(float(df_copy['processing_time'].mean()), 2)
                    operational_metrics['median_processing_time_hours'] = round(float(df_copy['processing_time'].median()), 2)
                    operational_metrics['fastest_order_hours'] = round(float(df_copy['processing_time'].min()), 2)
                    operational_metrics['slowest_order_hours'] = round(float(df_copy['processing_time'].max()), 2)
                    
                    # Efficiency metrics
                    fast_orders = len(df_copy[df_copy['processing_time'] <= 1])
                    normal_orders = len(df_copy[(df_copy['processing_time'] > 1) & (df_copy['processing_time'] <= 6)])
                    slow_orders = len(df_copy[df_copy['processing_time'] > 6])
                    
                    operational_metrics['fast_processing_rate'] = round((fast_orders / len(df_copy)) * 100, 2)
                    operational_metrics['normal_processing_rate'] = round((normal_orders / len(df_copy)) * 100, 2)
                    operational_metrics['slow_processing_rate'] = round((slow_orders / len(df_copy)) * 100, 2)
                    
            except Exception as e:
                logger.error("Error in processing time analysis: %s", str(e))
        
        metrics['operational_metrics'] = operational_metrics
        
        # 4. Customer Performance Metrics (Optimized)
        customer_metrics = {}
        
        if len(customer_counts) > 0:
            try:
                customer_metrics['total_customers'] = int(len(customer_counts))
                customer_metrics['average_orders_per_customer'] = round(customer_counts.mean(), 2)
                customer_metrics['max_orders_by_customer'] = int(customer_counts.max())
                customer_metrics['min_orders_by_customer'] = int(customer_counts.min())
                
                # Customer segmentation (optimized)
                high_value_customers = customer_counts[customer_counts >= customer_counts.quantile(0.8)]
                medium_value_customers = customer_counts[(customer_counts >= customer_counts.quantile(0.4)) & (customer_counts < customer_counts.quantile(0.8))]
                low_value_customers = customer_counts[customer_counts < customer_counts.quantile(0.4)]
                
                customer_metrics['high_value_customers'] = int(len(high_value_customers))
                customer_metrics['medium_value_customers'] = int(len(medium_value_customers))
                customer_metrics['low_value_customers'] = int(len(low_value_customers))
                
                # Top customers (optimized)
                customer_metrics['top_10_customers'] = {str(name): int(count) for name, count in customer_counts.head(10).items()}
            except Exception as e:
                logger.error("Error in customer metrics: %s", str(e))
                customer_metrics = {
                    'total_customers': 0,
                    'average_orders_per_customer': 0,
                    'max_orders_by_customer': 0,
                    'min_orders_by_customer': 0,
                    'high_value_customers': 0,
                    'medium_value_customers': 0,
                    'low_value_customers': 0,
                    'top_10_customers': {}
                }
        else:
            customer_metrics = {
                'total_customers': 0,
                'average_orders_per_customer': 0,
                'max_orders_by_customer': 0,
                'min_orders_by_customer': 0,
                'high_value_customers': 0,
                'medium_value_customers': 0,
                'low_value_customers': 0,
                'top_10_customers': {}
            }
        
        metrics['customer_metrics'] = customer_metrics
        
        # 5. Technician Performance Metrics (Optimized)
        technician_metrics = {}
        
        if len(technician_counts) > 0:
            try:
                technician_metrics['total_technicians'] = int(len(technician_counts))
                technician_metrics['average_orders_per_technician'] = round(technician_counts.mean(), 2)
                technician_metrics['max_orders_by_technician'] = int(technician_counts.max())
                technician_metrics['min_orders_by_technician'] = int(technician_counts.min())
                
                # Technician performance tiers (optimized)
                top_performers = technician_counts[technician_counts >= technician_counts.quantile(0.8)]
                avg_performers = technician_counts[(technician_counts >= technician_counts.quantile(0.4)) & (technician_counts < technician_counts.quantile(0.8))]
                low_performers = technician_counts[technician_counts < technician_counts.quantile(0.4)]
                
                technician_metrics['top_performers'] = int(len(top_performers))
                technician_metrics['average_performers'] = int(len(avg_performers))
                technician_metrics['low_performers'] = int(len(low_performers))
                
                # Top technicians (optimized)
                technician_metrics['top_10_technicians'] = {str(name): int(count) for name, count in technician_counts.head(10).items()}
            except Exception as e:
                logger.error("Error in technician metrics: %s", str(e))
                technician_metrics = {
                    'total_technicians': 0,
                    'average_orders_per_technician': 0,
                    'max_orders_by_technician': 0,
                    'min_orders_by_technician': 0,
                    'top_performers': 0,
                    'average_performers': 0,
                    'low_performers': 0,
                    'top_10_technicians': {}
                }
        else:
            technician_metrics = {
                'total_technicians': 0,
                'average_orders_per_technician': 0,
                'max_orders_by_technician': 0,
                'min_orders_by_technician': 0,
                'top_performers': 0,
                'average_performers': 0,
                'low_performers': 0,
                'top_10_technicians': {}
            }
        
        metrics['technician_metrics'] = technician_metrics
        
        # 6. Service Performance Metrics (Optimized)
        service_metrics = {}
        
        if len(service_counts) > 0:
            try:
                service_metrics['total_service_types'] = int(len(service_counts))
                service_metrics['most_popular_service'] = str(service_counts.index[0])
                service_metrics['most_popular_service_count'] = int(service_counts.iloc[0])
                service_metrics['service_distribution'] = {str(service): int(count) for service, count in service_counts.items()}
            except Exception as e:
                logger.error("Error in service metrics: %s", str(e))
                service_metrics = {
                    'total_service_types': 0,
                    'most_popular_service': 'N/A',
                    'most_popular_service_count': 0,
                    'service_distribution': {}
                }
        else:
            service_metrics = {
                'total_service_types': 0,
                'most_popular_service': 'N/A',
                'most_popular_service_count': 0,
                'service_distribution': {}
            }
        
        metrics['service_metrics'] = service_metrics
        
        # 7. Time-based Performance Metrics
        time_metrics = {}
        
        if 'Created At' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                df_copy = df_copy.dropna(subset=['Created At'])
                
                if len(df_copy) > 0:
                    df_copy['date'] = df_copy['Created At'].dt.date
                    daily_orders = df_copy.groupby('date').size()
                    
                    time_metrics['total_days'] = int(len(daily_orders))
                    time_metrics['average_orders_per_day'] = round(daily_orders.mean(), 2)
                    time_metrics['peak_day_orders'] = int(daily_orders.max())
                    time_metrics['lowest_day_orders'] = int(daily_orders.min())
                    time_metrics['peak_day'] = str(daily_orders.idxmax()) if len(daily_orders) > 0 else 'N/A'
                    time_metrics['lowest_day'] = str(daily_orders.idxmin()) if len(daily_orders) > 0 else 'N/A'
                    
                    # Monthly analysis
                    df_copy['month'] = df_copy['Created At'].dt.to_period('M')
                    monthly_orders = df_copy.groupby('month').size()
                    time_metrics['total_months'] = int(len(monthly_orders))
                    time_metrics['average_orders_per_month'] = round(monthly_orders.mean(), 2)
                    time_metrics['peak_month_orders'] = int(monthly_orders.max())
                    time_metrics['peak_month'] = str(monthly_orders.idxmax()) if len(monthly_orders) > 0 else 'N/A'
                else:
                    time_metrics = {
                        'total_days': 0,
                        'average_orders_per_day': 0,
                        'peak_day_orders': 0,
                        'lowest_day_orders': 0,
                        'peak_day': 'N/A',
                        'lowest_day': 'N/A',
                        'total_months': 0,
                        'average_orders_per_month': 0,
                        'peak_month_orders': 0,
                        'peak_month': 'N/A'
                    }
                    
            except Exception as e:
                logger.error("Error in time-based analysis: %s", str(e))
                time_metrics = {
                    'total_days': 0,
                    'average_orders_per_day': 0,
                    'peak_day_orders': 0,
                    'lowest_day_orders': 0,
                    'peak_day': 'N/A',
                    'lowest_day': 'N/A',
                    'total_months': 0,
                    'average_orders_per_month': 0,
                    'peak_month_orders': 0,
                    'peak_month': 'N/A'
                }
        
        metrics['time_metrics'] = time_metrics
        
        # 8. Performance Score and Recommendations
        performance_score = 0
        max_score = 100
        recommendations = []
        
        # Score based on completion rate
        completion_rate = operational_metrics.get('completion_rate', 0)
        if completion_rate >= 90:
            performance_score += 25
        elif completion_rate >= 80:
            performance_score += 20
        elif completion_rate >= 70:
            performance_score += 15
        else:
            performance_score += 5
            recommendations.append("📈 **Low Completion Rate**: Focus on improving order completion rates.")
        
        # Score based on processing efficiency
        fast_rate = operational_metrics.get('fast_processing_rate', 0)
        if fast_rate >= 60:
            performance_score += 25
        elif fast_rate >= 40:
            performance_score += 20
        elif fast_rate >= 20:
            performance_score += 15
        else:
            performance_score += 5
            recommendations.append("⏱️ **Slow Processing**: Optimize workflow to improve processing times.")
        
        # Score based on customer satisfaction (low cancellation rate)
        cancellation_rate = operational_metrics.get('cancellation_rate', 0)
        if cancellation_rate <= 5:
            performance_score += 25
        elif cancellation_rate <= 10:
            performance_score += 20
        elif cancellation_rate <= 15:
            performance_score += 15
        else:
            performance_score += 5
            recommendations.append("❌ **High Cancellations**: Investigate and address cancellation reasons.")
        
        # Score based on financial performance
        avg_order_value = financial_metrics.get('average_order_value', 0)
        if avg_order_value > 1000:
            performance_score += 25
        elif avg_order_value > 500:
            performance_score += 20
        elif avg_order_value > 100:
            performance_score += 15
        else:
            performance_score += 10
            recommendations.append("💰 **Low Average Order Value**: Focus on increasing order values.")
        
        performance_score = min(performance_score, max_score)
        
        # Performance level
        if performance_score >= 80:
            performance_level = "EXCELLENT"
        elif performance_score >= 60:
            performance_level = "GOOD"
        elif performance_score >= 40:
            performance_level = "AVERAGE"
        else:
            performance_level = "NEEDS IMPROVEMENT"
        
        if performance_score >= 80:
            recommendations.append("🎉 **Excellent Performance**: Keep up the great work!")
        elif performance_score >= 60:
            recommendations.append("✅ **Good Performance**: Continue optimizing for better results.")
        elif performance_score >= 40:
            recommendations.append("⚠️ **Average Performance**: Focus on key improvement areas.")
        else:
            recommendations.append("🚨 **Performance Needs Attention**: Prioritize improvement initiatives.")
        
        metrics['performance_score'] = performance_score
        metrics['performance_level'] = performance_level
        metrics['max_score'] = max_score
        metrics['recommendations'] = recommendations
        
        return metrics
    except Exception as e:
        logger.error("Error in performance metrics: %s", str(e))
        return {'error': str(e)}

def generate_trend_analysis(df, column_types):
    """Generate trend analysis over time"""
    try:
        trends = {}
        date_cols = column_types.get('date_columns', [])
        
        if date_cols:
            date_col = date_cols[0]  # Use first date column
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            
            if len(df_copy) > 0:
                # Monthly trends
                df_copy['month'] = df_copy[date_col].dt.to_period('M')
                monthly_trends = df_copy.groupby('month').size()
                trends['monthly_order_trends'] = {str(month): int(count) for month, count in monthly_trends.items()}
                
                # Weekly trends
                df_copy['week'] = df_copy[date_col].dt.to_period('W')
                weekly_trends = df_copy.groupby('week').size()
                trends['weekly_order_trends'] = {str(week): int(count) for week, count in weekly_trends.items()}
                
                # Revenue trends by month
                financial_cols = column_types.get('financial_columns', [])
                if financial_cols:
                    monthly_revenue = df_copy.groupby('month').agg({
                        col: 'sum' for col in financial_cols if col in df_copy.columns
                    }).sum(axis=1)
                    trends['monthly_revenue_trends'] = {str(month): float(revenue) for month, revenue in monthly_revenue.items()}
        
        return trends
    except Exception as e:
        logger.error("Error in trend analysis: %s", str(e))
        return {'error': str(e)}

def generate_risk_assessment(df, column_types):
    """Generate comprehensive risk assessment and anomaly detection"""
    try:
        risks = {}
        risk_score = 0
        max_risk_score = 100
        
        # 1. Data Quality Risk Assessment
        data_quality_risks = {}
        
        # Missing Data Risk (Weight: 25%)
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data.sum() / (len(df) * len(df.columns))) * 100
        data_quality_risks['missing_data_percentage'] = round(missing_percentage, 2)
        
        high_missing = missing_data[missing_data > len(df) * 0.1]  # More than 10% missing
        critical_missing = missing_data[missing_data > len(df) * 0.5]  # More than 50% missing
        data_quality_risks['high_missing_columns'] = {col: int(count) for col, count in high_missing.items()}
        data_quality_risks['critical_missing_columns'] = {col: int(count) for col, count in critical_missing.items()}
        
        if missing_percentage > 20:
            risk_score += 25
        elif missing_percentage > 10:
            risk_score += 15
        elif missing_percentage > 5:
            risk_score += 10
        
        # Duplicate Risk (Weight: 15%)
        duplicate_count = int(df.duplicated().sum())
        duplicate_percentage = round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0
        data_quality_risks['duplicate_records'] = duplicate_count
        data_quality_risks['duplicate_percentage'] = duplicate_percentage
        
        if duplicate_percentage > 10:
            risk_score += 15
        elif duplicate_percentage > 5:
            risk_score += 10
        elif duplicate_percentage > 1:
            risk_score += 5
        
        # 2. Financial Risk Assessment (Weight: 30%)
        financial_risks = {}
        financial_cols = column_types.get('financial_columns', [])
        
        for col in financial_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        # Outlier Detection
                        Q1 = numeric_data.quantile(0.25)
                        Q3 = numeric_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = numeric_data[(numeric_data < Q1 - 1.5 * IQR) | (numeric_data > Q3 + 1.5 * IQR)]
                        outlier_count = len(outliers)
                        
                        # Extreme Value Detection
                        extreme_outliers = numeric_data[(numeric_data < Q1 - 3 * IQR) | (numeric_data > Q3 + 3 * IQR)]
                        extreme_count = len(extreme_outliers)
                        
                        if outlier_count > 0:
                            financial_risks[col] = {
                                'total_outliers': int(outlier_count),
                                'extreme_outliers': int(extreme_count),
                                'outlier_percentage': round((outlier_count / len(numeric_data)) * 100, 2),
                                'mean_value': float(numeric_data.mean()),
                                'max_value': float(numeric_data.max()),
                                'min_value': float(numeric_data.min()),
                                'std_dev': float(numeric_data.std())
                            }
                            
                            # Risk scoring for financial outliers
                            outlier_percentage = (outlier_count / len(numeric_data)) * 100
                            if outlier_percentage > 10:
                                risk_score += 15
                            elif outlier_percentage > 5:
                                risk_score += 10
                            elif outlier_percentage > 1:
                                risk_score += 5
                        
                        # Zero/Null Value Risk
                        zero_count = len(numeric_data[numeric_data == 0])
                        zero_percentage = (zero_count / len(numeric_data)) * 100
                        if zero_percentage > 50:
                            financial_risks[col]['zero_value_risk'] = {
                                'zero_count': int(zero_count),
                                'zero_percentage': round(zero_percentage, 2)
                            }
                            risk_score += 5
                            
                except Exception as e:
                    logger.error("Error analyzing financial column %s: %s", safe_str(col), str(e))
                    continue
        
        # 3. Operational Risk Assessment (Weight: 20%)
        operational_risks = {}
        
        # Cancellation Risk
        if 'Order Status' in df.columns:
            cancelled_orders = df[df['Order Status'].str.contains('cancelled|cancel', case=False, na=False)]
            cancellation_rate = round((len(cancelled_orders) / len(df)) * 100, 2) if len(df) > 0 else 0
            operational_risks['cancellation_rate'] = cancellation_rate
            
            if cancellation_rate > 20:
                risk_score += 20
            elif cancellation_rate > 10:
                risk_score += 15
            elif cancellation_rate > 5:
                risk_score += 10
        
        # Processing Time Risk
        if 'Created At' in df.columns and 'Completed Date' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                
                if len(df_copy) > 0:
                    df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                    avg_processing_time = df_copy['processing_time'].mean()
                    max_processing_time = df_copy['processing_time'].max()
                    
                    operational_risks['processing_time_analysis'] = {
                        'average_hours': round(float(avg_processing_time), 2),
                        'max_hours': round(float(max_processing_time), 2),
                        'orders_over_24h': int(len(df_copy[df_copy['processing_time'] > 24])),
                        'orders_over_48h': int(len(df_copy[df_copy['processing_time'] > 48]))
                    }
                    
                    # Risk scoring for processing time
                    if avg_processing_time > 48:
                        risk_score += 15
                    elif avg_processing_time > 24:
                        risk_score += 10
                    elif avg_processing_time > 12:
                        risk_score += 5
                        
            except Exception as e:
                logger.error("Error in processing time analysis: %s", str(e))
        
        # 4. Customer Risk Assessment (Weight: 15%)
        customer_risks = {}
        
        if 'Customer Name' in df.columns:
            customer_orders = df.groupby('Customer Name').size()
            high_frequency_customers = customer_orders[customer_orders > customer_orders.quantile(0.95)]
            customer_risks['high_frequency_customers'] = {
                'count': int(len(high_frequency_customers)),
                'customers': {name: int(count) for name, count in high_frequency_customers.items()}
            }
            
            # Customer concentration risk
            top_10_percentage = (high_frequency_customers.sum() / len(df)) * 100
            customer_risks['customer_concentration'] = round(top_10_percentage, 2)
            
            if top_10_percentage > 50:
                risk_score += 15
            elif top_10_percentage > 30:
                risk_score += 10
            elif top_10_percentage > 20:
                risk_score += 5
        
        # 5. Geographic Risk Assessment (Weight: 10%)
        geographic_risks = {}
        location_cols = column_types.get('location_columns', [])
        
        for loc_col in location_cols[:2]:  # Analyze first 2 location columns
            if loc_col in df.columns:
                location_distribution = df[loc_col].value_counts()
                top_locations = location_distribution.head(5)
                concentration_percentage = (top_locations.sum() / len(df)) * 100
                
                geographic_risks[f'{loc_col}_concentration'] = {
                    'top_5_percentage': round(concentration_percentage, 2),
                    'top_locations': {loc: int(count) for loc, count in top_locations.items()}
                }
                
                if concentration_percentage > 80:
                    risk_score += 10
                elif concentration_percentage > 60:
                    risk_score += 7
                elif concentration_percentage > 40:
                    risk_score += 5
        
        # 6. Overall Risk Score and Recommendations
        risk_score = min(risk_score, max_risk_score)  # Cap at 100
        
        risk_level = "LOW"
        if risk_score >= 70:
            risk_level = "CRITICAL"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        
        # Generate recommendations
        recommendations = []
        if missing_percentage > 10:
            recommendations.append("🔍 **Data Quality**: High missing data detected. Consider data validation and collection improvements.")
        if duplicate_percentage > 5:
            recommendations.append("🔄 **Duplicates**: Significant duplicate records found. Implement deduplication processes.")
        if any('outlier_percentage' in risks.get('financial_outliers', {}).values() for risks in [financial_risks]):
            recommendations.append("💰 **Financial Anomalies**: Financial outliers detected. Review for potential fraud or errors.")
        if cancellation_rate > 10:
            recommendations.append("❌ **High Cancellations**: Elevated cancellation rate. Investigate operational issues.")
        if risk_score < 30:
            recommendations.append("✅ **Good Standing**: Overall data quality is acceptable. Continue monitoring.")
        
        risks = {
            'overall_risk_score': risk_score,
            'risk_level': risk_level,
            'max_risk_score': max_risk_score,
            'data_quality_risks': data_quality_risks,
            'financial_risks': financial_risks,
            'operational_risks': operational_risks,
            'customer_risks': customer_risks,
            'geographic_risks': geographic_risks,
            'recommendations': recommendations
        }
        
        return risks
    except Exception as e:
        logger.error("Error in risk assessment: %s", str(e))
        return {'error': str(e)}

def generate_customer_analytics(df, column_types):
    """Generate customer-focused analytics"""
    try:
        customer_insights = {}
        
        # Customer Segmentation
        if 'Customer Name' in df.columns:
            customer_orders = df.groupby('Customer Name').size().sort_values(ascending=False)
            customer_insights['top_customers'] = {name: int(count) for name, count in customer_orders.head(10).items()}
            customer_insights['total_unique_customers'] = int(customer_orders.nunique())
            customer_insights['average_orders_per_customer'] = round(len(df) / customer_orders.nunique(), 2) if customer_orders.nunique() > 0 else 0
        
        # Customer Value Analysis
        if 'Customer Name' in df.columns and 'Customer Value' in df.columns:
            try:
                customer_value = df.groupby('Customer Name')['Customer Value'].sum().sort_values(ascending=False)
                customer_insights['top_customer_values'] = {name: float(value) for name, value in customer_value.head(10).items()}
                customer_insights['average_customer_value'] = float(customer_value.mean()) if len(customer_value) > 0 else 0
            except:
                pass
        
        # Geographic Customer Distribution
        location_cols = column_types.get('location_columns', [])
        if location_cols:
            for loc_col in location_cols[:2]:  # Use first 2 location columns
                if loc_col in df.columns:
                    location_distribution = df[loc_col].value_counts().head(10)
                    customer_insights[f'top_{loc_col.lower().replace(" ", "_")}'] = {loc: int(count) for loc, count in location_distribution.items()}
        
        return customer_insights
    except Exception as e:
        logger.error("Error in customer analytics: %s", str(e))
        return {'error': str(e)}

def generate_operational_insights(df, column_types):
    """Generate comprehensive operational and efficiency insights"""
    try:
        operational = {}
        
        # 1. Technician Performance Analysis
        if 'Technician Name' in df.columns:
            tech_analysis = {}
            tech_performance = df.groupby('Technician Name').size().sort_values(ascending=False)
            
            # Basic metrics
            tech_analysis['total_technicians'] = int(tech_performance.nunique())
            tech_analysis['average_orders_per_technician'] = round(len(df) / tech_performance.nunique(), 2) if tech_performance.nunique() > 0 else 0
            tech_analysis['top_10_technicians'] = {name: int(count) for name, count in tech_performance.head(10).items()}
            tech_analysis['bottom_10_technicians'] = {name: int(count) for name, count in tech_performance.tail(10).items()}
            
            # Performance tiers
            top_performers = tech_performance[tech_performance >= tech_performance.quantile(0.8)]
            avg_performers = tech_performance[(tech_performance >= tech_performance.quantile(0.4)) & (tech_performance < tech_performance.quantile(0.8))]
            low_performers = tech_performance[tech_performance < tech_performance.quantile(0.4)]
            
            tech_analysis['performance_tiers'] = {
                'top_performers': int(len(top_performers)),
                'average_performers': int(len(avg_performers)),
                'low_performers': int(len(low_performers))
            }
            
            # Efficiency analysis with processing time
            if 'Created At' in df.columns and 'Completed Date' in df.columns:
                try:
                    df_copy = df.copy()
                    df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                    df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                    df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                    
                    if len(df_copy) > 0:
                        df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                        tech_efficiency = df_copy.groupby('Technician Name')['processing_time'].agg(['mean', 'count']).sort_values('mean')
                        
                        tech_analysis['most_efficient_technicians'] = {
                            name: {'avg_time': round(float(mean), 2), 'orders': int(count)} 
                            for name, (mean, count) in tech_efficiency.head(10).items()
                        }
                        tech_analysis['least_efficient_technicians'] = {
                            name: {'avg_time': round(float(mean), 2), 'orders': int(count)} 
                            for name, (mean, count) in tech_efficiency.tail(10).items()
                        }
                except Exception as e:
                    logger.error("Error in technician efficiency analysis: %s", str(e))
            
            operational['technician_analysis'] = tech_analysis
        
        # 2. Facility Performance Analysis
        if 'Facility Name' in df.columns:
            facility_analysis = {}
            facility_performance = df.groupby('Facility Name').size().sort_values(ascending=False)
            
            facility_analysis['total_facilities'] = int(facility_performance.nunique())
            facility_analysis['top_10_facilities'] = {name: int(count) for name, count in facility_performance.head(10).items()}
            facility_analysis['bottom_10_facilities'] = {name: int(count) for name, count in facility_performance.tail(10).items()}
            
            # Facility efficiency analysis
            if 'Created At' in df.columns and 'Completed Date' in df.columns:
                try:
                    df_copy = df.copy()
                    df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                    df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                    df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                    
                    if len(df_copy) > 0:
                        df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                        facility_efficiency = df_copy.groupby('Facility Name')['processing_time'].agg(['mean', 'count']).sort_values('mean')
                        
                        facility_analysis['most_efficient_facilities'] = {
                            name: {'avg_time': round(float(mean), 2), 'orders': int(count)} 
                            for name, (mean, count) in facility_efficiency.head(10).items()
                        }
                        facility_analysis['least_efficient_facilities'] = {
                            name: {'avg_time': round(float(mean), 2), 'orders': int(count)} 
                            for name, (mean, count) in facility_efficiency.tail(10).items()
                        }
                except Exception as e:
                    logger.error("Error in facility efficiency analysis: %s", str(e))
            
            operational['facility_analysis'] = facility_analysis
        
        # 3. Service Type Analysis
        if 'Service Type' in df.columns:
            service_analysis = {}
            service_distribution = df['Service Type'].value_counts()
            service_analysis['distribution'] = {service: int(count) for service, count in service_distribution.items()}
            service_analysis['total_service_types'] = int(len(service_distribution))
            
            # Service efficiency analysis
            if 'Created At' in df.columns and 'Completed Date' in df.columns:
                try:
                    df_copy = df.copy()
                    df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                    df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                    df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                    
                    if len(df_copy) > 0:
                        df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                        service_efficiency = df_copy.groupby('Service Type')['processing_time'].agg(['mean', 'count']).sort_values('mean')
                        
                        service_analysis['service_efficiency'] = {
                            service: {'avg_time': round(float(mean), 2), 'orders': int(count)} 
                            for service, (mean, count) in service_efficiency.items()
                        }
                except Exception as e:
                    logger.error("Error in service efficiency analysis: %s", str(e))
            
            operational['service_analysis'] = service_analysis
        
        # 4. Order Status and Workflow Analysis
        if 'Order Status' in df.columns:
            status_analysis = {}
            status_distribution = df['Order Status'].value_counts()
            status_analysis['distribution'] = {status: int(count) for status, count in status_distribution.items()}
            status_analysis['total_statuses'] = int(len(status_distribution))
            
            # Status efficiency analysis
            if 'Created At' in df.columns and 'Completed Date' in df.columns:
                try:
                    df_copy = df.copy()
                    df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                    df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                    df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                    
                    if len(df_copy) > 0:
                        df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                        status_efficiency = df_copy.groupby('Order Status')['processing_time'].agg(['mean', 'count']).sort_values('mean')
                        
                        status_analysis['status_efficiency'] = {
                            status: {'avg_time': round(float(mean), 2), 'orders': int(count)} 
                            for status, (mean, count) in status_efficiency.items()
                        }
                except Exception as e:
                    logger.error("Error in status efficiency analysis: %s", str(e))
            
            operational['status_analysis'] = status_analysis
        
        # 5. Comprehensive Efficiency Metrics
        efficiency_metrics = {}
        
        if 'Created At' in df.columns and 'Completed Date' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                df_copy['Completed Date'] = pd.to_datetime(df_copy['Completed Date'], errors='coerce')
                df_copy = df_copy.dropna(subset=['Created At', 'Completed Date'])
                
                if len(df_copy) > 0:
                    df_copy['processing_time'] = (df_copy['Completed Date'] - df_copy['Created At']).dt.total_seconds() / 3600
                    
                    efficiency_metrics['processing_time_analysis'] = {
                        'average_hours': round(float(df_copy['processing_time'].mean()), 2),
                        'median_hours': round(float(df_copy['processing_time'].median()), 2),
                        'min_hours': round(float(df_copy['processing_time'].min()), 2),
                        'max_hours': round(float(df_copy['processing_time'].max()), 2),
                        'std_dev_hours': round(float(df_copy['processing_time'].std()), 2),
                        'orders_under_1h': int(len(df_copy[df_copy['processing_time'] <= 1])),
                        'orders_1_6h': int(len(df_copy[(df_copy['processing_time'] > 1) & (df_copy['processing_time'] <= 6)])),
                        'orders_6_24h': int(len(df_copy[(df_copy['processing_time'] > 6) & (df_copy['processing_time'] <= 24)])),
                        'orders_over_24h': int(len(df_copy[df_copy['processing_time'] > 24])),
                        'orders_over_48h': int(len(df_copy[df_copy['processing_time'] > 48])),
                        'total_processed_orders': int(len(df_copy))
                    }
                    
                    # Efficiency benchmarks
                    total_orders = len(df_copy)
                    if total_orders > 0:
                        efficiency_metrics['efficiency_benchmarks'] = {
                            'fast_processing_rate': round((efficiency_metrics['processing_time_analysis']['orders_under_1h'] / total_orders) * 100, 2),
                            'normal_processing_rate': round((efficiency_metrics['processing_time_analysis']['orders_1_6h'] / total_orders) * 100, 2),
                            'slow_processing_rate': round((efficiency_metrics['processing_time_analysis']['orders_over_24h'] / total_orders) * 100, 2)
                        }
                    
            except Exception as e:
                logger.error("Error in efficiency metrics: %s", str(e))
        
        operational['efficiency_metrics'] = efficiency_metrics
        
        # 6. Operational KPIs
        operational_kpis = {}
        
        # Completion rate
        if 'Order Status' in df.columns:
            completed_orders = df[df['Order Status'].str.contains('completed|done|finished', case=False, na=False)]
            operational_kpis['completion_rate'] = round((len(completed_orders) / len(df)) * 100, 2) if len(df) > 0 else 0
        
        # Cancellation rate
        if 'Order Status' in df.columns:
            cancelled_orders = df[df['Order Status'].str.contains('cancelled|cancel', case=False, na=False)]
            operational_kpis['cancellation_rate'] = round((len(cancelled_orders) / len(df)) * 100, 2) if len(df) > 0 else 0
        
        # Average orders per day (if date available)
        if 'Created At' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['Created At'] = pd.to_datetime(df_copy['Created At'], errors='coerce')
                df_copy = df_copy.dropna(subset=['Created At'])
                if len(df_copy) > 0:
                    df_copy['date'] = df_copy['Created At'].dt.date
                    daily_orders = df_copy.groupby('date').size()
                    operational_kpis['average_orders_per_day'] = round(daily_orders.mean(), 2)
                    operational_kpis['peak_day_orders'] = int(daily_orders.max())
                    operational_kpis['total_days'] = int(len(daily_orders))
            except Exception as e:
                logger.error("Error in daily order analysis: %s", str(e))
        
        operational['operational_kpis'] = operational_kpis
        
        # 7. Operational Recommendations
        recommendations = []
        
        if 'technician_analysis' in operational:
            tech_analysis = operational['technician_analysis']
            if tech_analysis.get('performance_tiers', {}).get('low_performers', 0) > 0:
                recommendations.append("👥 **Technician Performance**: Consider training or support for low-performing technicians.")
            if tech_analysis.get('average_orders_per_technician', 0) < 10:
                recommendations.append("📊 **Workload Distribution**: Consider redistributing workload for better efficiency.")
        
        if 'efficiency_metrics' in operational:
            eff_metrics = operational['efficiency_metrics']
            if 'efficiency_benchmarks' in eff_metrics:
                slow_rate = eff_metrics['efficiency_benchmarks'].get('slow_processing_rate', 0)
                if slow_rate > 20:
                    recommendations.append("⏱️ **Processing Time**: High number of slow orders. Review workflow bottlenecks.")
        
        if 'operational_kpis' in operational:
            kpis = operational['operational_kpis']
            if kpis.get('cancellation_rate', 0) > 10:
                recommendations.append("❌ **High Cancellations**: Investigate reasons for elevated cancellation rate.")
            if kpis.get('completion_rate', 0) < 80:
                recommendations.append("✅ **Completion Rate**: Focus on improving order completion rates.")
        
        operational['recommendations'] = recommendations
        
        return operational
    except Exception as e:
        logger.error("Error in operational insights: %s", str(e))
        return {'error': str(e)}

def generate_financial_deep_dive(df, column_types):
    """Generate detailed financial analysis"""
    try:
        financial = {}
        
        financial_cols = column_types.get('financial_columns', [])
        
        for col in financial_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        financial[col] = {
                            'total': float(numeric_data.sum()),
                            'average': float(numeric_data.mean()),
                            'median': float(numeric_data.median()),
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'std_dev': float(numeric_data.std()),
                            'quartiles': {
                                'q1': float(numeric_data.quantile(0.25)),
                                'q2': float(numeric_data.quantile(0.50)),
                                'q3': float(numeric_data.quantile(0.75))
                            },
                            'top_10_values': [float(x) for x in numeric_data.nlargest(10).tolist()],
                            'bottom_10_values': [float(x) for x in numeric_data.nsmallest(10).tolist()]
                        }
                        
                        # Distribution analysis
                        bins = pd.cut(numeric_data, bins=10)
                        distribution = bins.value_counts().sort_index()
                        financial[col]['distribution'] = {str(interval): int(count) for interval, count in distribution.items()}
                        
                except Exception as e:
                    logger.error("Error analyzing financial column %s: %s", safe_str(col), str(e))
                    continue
        
        # Cross-column analysis
        if len(financial_cols) >= 2:
            correlations = {}
            for i, col1 in enumerate(financial_cols):
                if col1 in df.columns:
                    for col2 in financial_cols[i+1:]:
                        if col2 in df.columns:
                            try:
                                data1 = pd.to_numeric(df[col1], errors='coerce')
                                data2 = pd.to_numeric(df[col2], errors='coerce')
                                correlation = data1.corr(data2)
                                if not pd.isna(correlation):
                                    correlations[f"{col1}_vs_{col2}"] = round(float(correlation), 3)
                            except:
                                continue
            financial['correlations'] = correlations
        
        return financial
    except Exception as e:
        logger.error("Error in financial deep dive: %s", str(e))
        return {'error': str(e)}

def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"File uploaded: {filename}")
            
            # Read the Excel file
            try:
                df = pd.read_excel(filepath, engine='openpyxl')
                logger.info(f"Successfully read Excel file with {len(df)} rows and {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error reading Excel file: {str(e)}")
                return jsonify({'error': f'Error reading Excel file: {str(e)}'}), 500
        
                    # Generate analytics
            basic_analytics = generate_basic_analytics(df)
            column_types = identify_column_types(df)
            revenue_insights = generate_revenue_insights(df, column_types)
            advanced_analytics = generate_advanced_analytics(df, column_types)
            
            # Add debugging info
            logger.info(f"Detected columns: {column_types}")
            logger.info(f"Location columns found: {column_types.get('location_columns', [])}")
            logger.info(f"Financial columns found: {column_types.get('financial_columns', [])}")
            logger.info(f"Revenue columns found: {column_types.get('revenue_columns', [])}")
            
            # Store data in session or return directly
            response_data = {
                'filename': filename,
                'basic_analytics': basic_analytics,
                'column_types': column_types,
                'revenue_insights': revenue_insights,
                'advanced_analytics': advanced_analytics,
                'columns': [str(col) for col in df.columns],
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns))
            }
            
            response_data = clean_nans(response_data)
            return Response(json.dumps(response_data, cls=NumpyEncoder, allow_nan=False), status=200, mimetype='application/json')
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload Excel files only.'}), 400
        
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/filter', methods=['POST'])
def filter_data():
    try:
        # Get filter parameters from request
        params = request.get_json()
        date_from = params.get('date_from')
        date_to = params.get('date_to')
        pincode = params.get('pincode')
        city = params.get('city')

        # Find the most recent uploaded Excel file
        upload_folder = app.config['UPLOAD_FOLDER']
        excel_files = [f for f in os.listdir(upload_folder) if f.endswith(('.xlsx', '.xls'))]
        if not excel_files:
            return jsonify({'error': 'No Excel files found in upload folder'}), 400
        latest_file = max(excel_files, key=lambda x: os.path.getctime(os.path.join(upload_folder, x)))
        filepath = os.path.join(upload_folder, latest_file)

        # Read the Excel file
        df = pd.read_excel(filepath, engine='openpyxl')
        original_len = len(df)

        # Identify columns
        column_types = identify_column_types(df)
        date_cols = column_types.get('date_columns', [])
        location_cols = column_types.get('location_columns', [])

        # Apply date filter (if provided)
        if date_from or date_to:
            if date_cols:
                date_col = date_cols[0]
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                if date_from:
                    df = df[df[date_col] >= pd.to_datetime(date_from)]
                if date_to:
                    df = df[df[date_col] <= pd.to_datetime(date_to)]

        # Apply pincode filter (if provided)
        if pincode and location_cols:
            # Try to find a pincode column
            pincode_col = next((col for col in location_cols if 'pincode' in col.lower() or 'zipcode' in col.lower() or 'pin' in col.lower()), None)
            if pincode_col:
                df = df[df[pincode_col].astype(str) == str(pincode)]

        # Apply city filter (if provided)
        if city and location_cols:
            city_col = next((col for col in location_cols if 'city' in col.lower()), None)
            if city_col:
                df = df[df[city_col].astype(str).str.lower() == str(city).lower()]

        # Prepare filtered raw data (limit to 1000 rows for performance)
        filtered_rows = df.head(1000).to_dict(orient='records')

        # Generate analytics on filtered data
        basic_analytics = generate_basic_analytics(df)
        revenue_insights = generate_revenue_insights(df, column_types)
        advanced_analytics = generate_advanced_analytics(df, column_types)

        response_data = {
            'filename': latest_file,
            'original_row_count': original_len,
            'filtered_row_count': len(df),
            'filters_applied': {
                'date_from': date_from,
                'date_to': date_to,
                'pincode': pincode,
                'city': city
            },
            'filtered_rows': filtered_rows,
            'basic_analytics': basic_analytics,
            'revenue_insights': revenue_insights,
            'advanced_analytics': advanced_analytics,
            'columns': [str(col) for col in df.columns],
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns))
        }
        response_data = clean_nans(response_data)
        return Response(json.dumps(response_data, cls=NumpyEncoder, allow_nan=False), status=200, mimetype='application/json')
    except Exception as e:
        logger.error(f"Error in filter: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/column/<column_name>')
def get_column_insights(column_name):
    try:
        # This would need to be implemented with session storage or database
        # For now, return a placeholder
        return jsonify({'message': f'Column insights for {column_name} would be generated here'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/columns')
def debug_columns():
    """Debug endpoint to see what columns are detected"""
    try:
        # Get the most recent uploaded file
        upload_folder = app.config['UPLOAD_FOLDER']
        excel_files = [f for f in os.listdir(upload_folder) if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            return jsonify({'error': 'No Excel files found in upload folder'})
        
        # Use the most recent file
        latest_file = max(excel_files, key=lambda x: os.path.getctime(os.path.join(upload_folder, x)))
        filepath = os.path.join(upload_folder, latest_file)
        
        df = pd.read_excel(filepath, engine='openpyxl')
        column_types = identify_column_types(df)
        
        return jsonify({
            'filename': latest_file,
            'all_columns': [str(col) for col in df.columns],
            'column_types': column_types,
            'sample_data': {
                str(col): df[col].dropna().head(3).tolist() for col in df.columns[:5]  # First 5 columns
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
