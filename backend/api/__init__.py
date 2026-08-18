from flask import Blueprint
from backend.api.upload import upload_bp
from backend.api.analysis import analysis_bp
from backend.api.auth import auth_bp
from backend.api.monitor import monitor_bp

api_bp = Blueprint('api', __name__, url_prefix='/api')
api_bp.register_blueprint(upload_bp)
api_bp.register_blueprint(analysis_bp)
api_bp.register_blueprint(auth_bp)
api_bp.register_blueprint(monitor_bp, url_prefix='/monitor')


