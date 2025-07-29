#!/usr/bin/env python3
"""
Enterprise Data Analytics Platform - Startup Script
"""

import os
import sys
from app import app
from config import config

def main():
    """Main startup function"""
    
    print("🚀 Enterprise Data Analytics Platform")
    print("=" * 50)
    
    # Get configuration from environment
    config_name = os.environ.get('FLASK_CONFIG') or 'default'
    app.config.from_object(config[config_name])
    
    # Initialize app with configuration
    config[config_name].init_app(app)
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Get host from environment or use default
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"📊 Configuration: {config_name}")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Processed folder: {app.config['PROCESSED_FOLDER']}")
    print(f"📏 Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print("=" * 50)
    
    try:
        print("🚀 Starting server...")
        app.run(
            host=host,
            port=port,
            debug=app.config['DEBUG'],
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 