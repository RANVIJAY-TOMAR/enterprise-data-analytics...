import os
from pathlib import Path

class Config:
    """Configuration settings for the Enterprise Data Analytics Platform"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
    
    # Data Processing Configuration
    CHUNK_SIZE = 10000  # Number of rows to process at once for large files
    MAX_ROWS_FOR_PANDAS = 100000  # Use Dask for files larger than this
    
    # Analytics Configuration
    MISSING_VALUE_THRESHOLD = 0.1  # 10% missing values threshold for risk alerts
    OUTLIER_MULTIPLIER = 1.5  # IQR multiplier for outlier detection
    
    # Chart Configuration
    MAX_CATEGORIES_DISPLAY = 10  # Maximum categories to show in charts
    CHART_HEIGHT = 400  # Default chart height in pixels
    
    # Performance Configuration
    ENABLE_CACHING = True
    CACHE_TIMEOUT = 3600  # 1 hour cache timeout
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'analytics_platform.log'
    
    # Database Configuration (for future use)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///analytics.db'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        
        # Create necessary directories
        Path(Config.UPLOAD_FOLDER).mkdir(exist_ok=True)
        Path(Config.PROCESSED_FOLDER).mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        # Set Flask configuration
        app.config['SECRET_KEY'] = Config.SECRET_KEY
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['PROCESSED_FOLDER'] = Config.PROCESSED_FOLDER
        app.config['ALLOWED_EXTENSIONS'] = Config.ALLOWED_EXTENSIONS

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB for production
    ENABLE_CACHING = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    UPLOAD_FOLDER = 'test_uploads'
    PROCESSED_FOLDER = 'test_processed'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 