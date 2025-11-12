#!/usr/bin/env python3
"""
Entry point cho Form Agent AI Backend
"""

import os
import sys
from app import create_app
from app.utils.helpers import setup_logging

# Setup logging
setup_logging()

# Create Flask app
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    host = app.config.get('HOST', '0.0.0.0')
    port = app.config.get('PORT', 8000)
    debug = app.config.get('DEBUG', True)
    
    print(f"🚀 Starting Form Agent AI Backend")
    print(f"📋 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs/")
    print(f"🔧 Debug Mode: {debug}")
    
    # Run the app
    app.run(
        host=host,
        port=port,
        debug=debug
    )