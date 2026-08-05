from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import re
from backend.core.database import create_user, get_user_by_email

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

EMAIL_REGEX = re.compile(r'^[^@]+@[^@]+\.[^@]+$')

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({
            'status': 'error',
            'error': 'All fields (name, email, password) are required.'
        }), 400

    if not EMAIL_REGEX.match(email):
        return jsonify({
            'status': 'error',
            'error': 'Invalid email format.'
        }), 400

    if len(password) < 8:
        return jsonify({
            'status': 'error',
            'error': 'Password must be at least 8 characters long.'
        }), 400


    # Check if duplicate email
    existing_user = get_user_by_email(email)
    if existing_user:
        return jsonify({
            'status': 'error',
            'error': 'Email is already registered.'
        }), 400

    # Hash the password
    password_hash = generate_password_hash(password)

    # Save user
    user_id = create_user(name, email, password_hash)
    if user_id is None:
        return jsonify({
            'status': 'error',
            'error': 'Failed to create user. Please try again.'
        }), 500

    return jsonify({
        'status': 'success',
        'message': 'Account created successfully. Please sign in.',
        'user': {
            'id': user_id,
            'name': name,
            'email': email
        }
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({
            'status': 'error',
            'error': 'Email and password are required.'
        }), 400

    user = get_user_by_email(email)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({
            'status': 'error',
            'error': 'Invalid email or password.'
        }), 401

    return jsonify({
        'status': 'success',
        'message': 'Login successful.',
        'user': {
            'id': user['id'],
            'name': user['name'],
            'email': user['email']
        }
    }), 200
