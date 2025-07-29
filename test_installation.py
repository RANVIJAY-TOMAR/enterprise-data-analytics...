#!/usr/bin/env python3
"""
Test script to verify the Enterprise Data Analytics Platform installation
"""

import sys
import importlib
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    
    required_packages = [
        'flask',
        'pandas',
        'dask',
        'openpyxl',
        'numpy',
        'plotly',
        'flask_cors',
        'werkzeug',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_directories():
    """Test if required directories exist or can be created"""
    
    required_dirs = [
        'uploads',
        'processed',
        'templates',
        'logs'
    ]
    
    print("\n📁 Testing directory structure...")
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"✅ {dir_name}/ (created)")
            except Exception as e:
                print(f"❌ {dir_name}/: {e}")
                missing_dirs.append(dir_name)
    
    return missing_dirs

def test_files():
    """Test if required files exist"""
    
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'templates/index.html',
        'README.md'
    ]
    
    print("\n📄 Testing required files...")
    missing_files = []
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    return missing_files

def test_flask_app():
    """Test if Flask app can be imported and configured"""
    
    print("\n🚀 Testing Flask application...")
    
    try:
        # Import app
        from app import app
        print("✅ Flask app imported successfully")
        
        # Test basic configuration
        if hasattr(app, 'config'):
            print("✅ App configuration available")
        else:
            print("❌ App configuration missing")
            return False
        
        # Test routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/upload', '/analytics/<filename>', '/files']
        
        for route in expected_routes:
            if route in routes or any(route.replace('<filename>', 'test') in r for r in routes):
                print(f"✅ Route {route}")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_data_processing():
    """Test basic data processing capabilities"""
    
    print("\n📊 Testing data processing capabilities...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'Order_ID': ['ORD001', 'ORD002', 'ORD003'],
            'Customer_Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Amount': [1000, 1500, 2000],
            'Rating': [4, 5, 3],
            'Date': pd.date_range('2023-01-01', periods=3)
        })
        
        print("✅ Test data created")
        
        # Test basic operations
        assert len(test_data) == 3, "Data length test failed"
        assert test_data['Amount'].sum() == 4500, "Sum calculation test failed"
        assert test_data['Rating'].mean() == 4.0, "Mean calculation test failed"
        
        print("✅ Basic data operations working")
        
        # Test Excel writing
        test_file = 'test_output.xlsx'
        test_data.to_excel(test_file, index=False)
        
        if Path(test_file).exists():
            print("✅ Excel file writing working")
            # Clean up
            Path(test_file).unlink()
        else:
            print("❌ Excel file writing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Enterprise Data Analytics Platform - Installation Test")
    print("=" * 60)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8 or higher recommended")
    
    # Run all tests
    failed_imports = test_imports()
    missing_dirs = test_directories()
    missing_files = test_files()
    flask_ok = test_flask_app()
    data_processing_ok = test_data_processing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    if not failed_imports and not missing_dirs and not missing_files and flask_ok and data_processing_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The Enterprise Data Analytics Platform is ready to use!")
        print("\n🚀 To start the platform, run:")
        print("   python app.py")
        print("   or")
        print("   python run.py")
        print("\n🌐 Then open your browser to: http://localhost:5000")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        
        if failed_imports:
            print(f"\n📦 Missing packages: {', '.join(failed_imports)}")
            print("   Run: pip install -r requirements.txt")
        
        if missing_dirs:
            print(f"\n📁 Missing directories: {', '.join(missing_dirs)}")
        
        if missing_files:
            print(f"\n📄 Missing files: {', '.join(missing_files)}")
        
        if not flask_ok:
            print("\n🚀 Flask app configuration issues detected")
        
        if not data_processing_ok:
            print("\n📊 Data processing capabilities not working")
        
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 