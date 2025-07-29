# Enterprise Data Analytics Platform

A powerful web-based data analytics platform designed for enterprise use, capable of processing and analyzing large-scale Excel datasets (up to 500MB per file) containing over 60 unique data fields related to service orders, customer behavior, logistics, and technician performance.

## 🚀 Features

### 📊 **Comprehensive Analytics**
- **File Upload & Parsing**: Drag-and-drop Excel file upload with support for .xlsx and .xls files up to 500MB
- **Data Processing**: Intelligent data cleaning, structuring, and validation using Python
- **Full-Spectrum Analysis**: Analyze all 60+ columns including:
  - Order details and financials
  - Service types and technician ratings
  - Customer zones and pincodes
  - Completion delays and extra billing
  - Operational metadata

### 📈 **Advanced Insights**
- **Trend Analysis**: Identify rising/declining zones, products, and services
- **Performance Metrics**: Track technician performance and customer satisfaction
- **Risk Detection**: Find red flags like missing values, delays, and repeated issues
- **Pattern Recognition**: Discover patterns across time and location

### 🎨 **Interactive Dashboard**
- **Modern UI**: Clean, professional Bootstrap-based interface
- **Interactive Charts**: Line, bar, pie charts with Chart.js and Plotly
- **Filterable Tables**: Sort and filter data for deeper analysis
- **KPI Cards**: Summary cards for key metrics (top city, technician, revenue trends)

### ⚡ **Scalability & Performance**
- **Large File Support**: Efficiently handle multiple large files without crashing
- **Modular Backend**: Flask + Pandas/Dask architecture
- **Memory Optimization**: Dask integration for processing large datasets
- **Real-time Processing**: Fast upload and analysis pipeline

## 🛠️ Technical Stack

- **Backend**: Python Flask
- **Data Processing**: Pandas + Dask for large file optimization
- **Frontend**: HTML + Bootstrap + Chart.js/Plotly
- **File Handling**: openpyxl for Excel support
- **Visualization**: Plotly for interactive charts

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd gigforce
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the platform**
   - Open your browser and go to `http://localhost:5000`
   - The platform will be ready to use!

## 📋 Usage Guide

### 1. **Upload Data**
- Navigate to the "Upload Data" section
- Drag and drop your Excel file or click to browse
- Supported formats: .xlsx, .xls (up to 500MB)
- The system will automatically process and analyze your data

### 2. **View Dashboard**
- **Overview**: See key metrics at a glance
- **Trends**: Analyze monthly/weekly patterns
- **Performance**: Track technician and service performance
- **Insights**: Get automated business insights

### 3. **Detailed Analytics**
- **Basic Stats**: Total records, columns, missing values, duplicates
- **Performance Metrics**: Ratings, revenue, financial analysis
- **Trend Analysis**: Location and service trends
- **Risk Assessment**: Data quality issues and potential problems

### 4. **File Management**
- View all uploaded files
- Access historical analytics
- Compare different datasets

## 🔍 Analytics Capabilities

### **Data Processing**
- Automatic data type detection and conversion
- Missing value analysis and reporting
- Duplicate record identification
- Outlier detection for numeric fields

### **Business Intelligence**
- **Customer Analysis**: Zone-wise performance, satisfaction ratings
- **Operational Insights**: Service completion rates, delays, cancellations
- **Financial Tracking**: Revenue trends, billing analysis, cost optimization
- **Performance Monitoring**: Technician efficiency, service quality metrics

### **Risk Management**
- **Data Quality**: Missing values, inconsistent data formats
- **Operational Risks**: Delays, cancellations, repeated issues
- **Financial Risks**: Billing discrepancies, revenue losses
- **Performance Risks**: Low-rated technicians, declining services

## 📊 Sample Data Fields Supported

The platform automatically detects and analyzes various data fields including:

- **Order Information**: Order ID, date, status, type
- **Customer Data**: Customer ID, name, contact, location
- **Service Details**: Service type, category, description
- **Technician Info**: Technician ID, name, rating, performance
- **Financial Data**: Amount, price, cost, billing details
- **Location Data**: Zone, city, pincode, area
- **Timestamps**: Created, updated, completed dates
- **Quality Metrics**: Ratings, satisfaction scores, feedback

## 🚀 Performance Features

### **Large File Handling**
- **Dask Integration**: Efficient processing of files up to 500MB
- **Memory Optimization**: Smart data loading and processing
- **Progress Tracking**: Real-time upload and processing status
- **Error Handling**: Robust error management and recovery

### **Scalability**
- **Modular Architecture**: Easy to extend and customize
- **API Endpoints**: RESTful API for integration
- **File Management**: Persistent storage and retrieval
- **Multi-file Support**: Handle multiple datasets simultaneously

## 🔧 Configuration

### **File Size Limits**
- Maximum file size: 500MB (configurable in `app.py`)
- Supported formats: .xlsx, .xls

### **Processing Options**
- Automatic data type detection
- Configurable missing value thresholds
- Customizable outlier detection parameters

### **Storage**
- Upload directory: `uploads/`
- Processed data: `processed/`
- Automatic cleanup and management

## 📈 Business Value

This platform enables companies to:

✅ **Upload raw operational data** and instantly extract actionable insights  
✅ **Identify trends** in order patterns, customer behavior, and technician performance  
✅ **Make data-backed decisions** without needing a data scientist  
✅ **Optimize operations** based on comprehensive analytics  
✅ **Reduce risks** through automated issue detection  
✅ **Improve customer satisfaction** through performance monitoring  

## 🤝 Support

For technical support or feature requests:
- Check the documentation
- Review error logs in the console
- Ensure all dependencies are properly installed

## 📄 License

This project is designed for enterprise use and can be customized according to specific business requirements.

---

**Ready to transform your data into actionable insights? Start analyzing your enterprise data today!** 🚀 