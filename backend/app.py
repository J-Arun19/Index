import os
from flask import Flask, jsonify
from flask_cors import CORS
from backend.config import Config
from backend.api import api_bp
from backend.core.database import init_db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize database
    init_db()

    
    # Enable CORS for all routes under /api/
    CORS(app, resources={r"/api/*": {"origins": "*"}}, expose_headers=["Content-Disposition"])
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Health check route
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'app': 'SatyaDrishti Backend API'
        }), 200
        
    # Root route
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            'message': 'Welcome to SatyaDrishti API. Use /api/upload to upload files.'
        }), 200

    # Analysis root route alias
    @app.route('/analysis', methods=['POST'])
    def analysis_root():
        from backend.api.analysis import analyze_media
        return analyze_media()
        
    # Error Handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        return jsonify({
            'status': 'error',
            'error': f'File is too large. Maximum allowed size is {max_mb:.0f} MB.'
        }), 413
        
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'status': 'error',
            'error': str(error.description or 'Bad request')
        }), 400
        
    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            'status': 'error',
            'error': 'Internal server error occurred'
        }), 500
        
    return app
