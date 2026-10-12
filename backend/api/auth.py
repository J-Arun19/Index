import logging
from flask import Blueprint, request, jsonify
import re
from backend.services.supabase_service import SupabaseService

logger = logging.getLogger("AuthAPI")
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

    if len(password) < 6:
        return jsonify({
            'status': 'error',
            'error': 'Password must be at least 6 characters long.'
        }), 400

    try:
        client = SupabaseService.get_client()
        res_auth = client.auth.admin.create_user({
            "email": email,
            "password": password,
            "user_metadata": {"full_name": name},
            "email_confirm": True
        })
        user = res_auth.user
        logger.info(f"Supabase Auth created user: {email} (ID: {user.id})")
        return jsonify({
            'status': 'success',
            'message': 'Account created successfully. Please sign in.',
            'user': {
                'id': user.id,
                'name': name,
                'email': email
            }
        }), 201
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Supabase Auth registration failed: {err_msg}")
        if "already registered" in err_msg.lower() or "unique" in err_msg.lower() or "user_already_exists" in err_msg.lower():
            return jsonify({
                'status': 'error',
                'error': 'Email is already registered.'
            }), 400
        return jsonify({
            'status': 'error',
            'error': f'Registration failed: {err_msg}'
        }), 500

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

    try:
        # Use an isolated client to avoid mutating the global SupabaseService service_role client singleton
        import os
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        auth_client = create_client(supabase_url, supabase_key)

        auth_res = auth_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        user = auth_res.user
        name = user.user_metadata.get('full_name') if user.user_metadata else email.split('@')[0]
        return jsonify({
            'status': 'success',
            'message': 'Login successful.',
            'user': {
                'id': user.id,
                'name': name,
                'email': user.email
            },
            'session': {
                'access_token': auth_res.session.access_token if auth_res.session else None
            }
        }), 200
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Supabase Auth login failed for masked user: {err_msg}")
        if "email not confirmed" in err_msg.lower() or "email_not_confirmed" in err_msg.lower():
            return jsonify({
                'status': 'error',
                'error': 'Please verify your email address before signing in.'
            }), 400
        if "invalid login credentials" in err_msg.lower() or "invalid_credentials" in err_msg.lower():
            return jsonify({
                'status': 'error',
                'error': 'Invalid email or password.'
            }), 401
        return jsonify({
            'status': 'error',
            'error': err_msg or 'Authentication failed.'
        }), 401

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json() or {}
    email = data.get('email', '').strip()
    redirect_to = data.get('redirectTo') or 'http://localhost:8080/reset-password'

    if not email:
        return jsonify({
            'status': 'error',
            'error': 'Email is required.'
        }), 400

    if not EMAIL_REGEX.match(email):
        return jsonify({
            'status': 'error',
            'error': 'Invalid email format.'
        }), 400

    try:
        client = SupabaseService.get_client()
        link_res = client.auth.admin.generate_link({
            'type': 'recovery',
            'email': email,
            'options': {
                'redirect_to': redirect_to
            }
        })

        action_link = None
        if hasattr(link_res, 'properties') and hasattr(link_res.properties, 'action_link'):
            action_link = link_res.properties.action_link
        elif isinstance(link_res, dict) and 'properties' in link_res:
            action_link = link_res['properties'].get('action_link')

        logger.info(f"Password recovery token generated for: {email}")

        return jsonify({
            'status': 'success',
            'message': 'Password recovery instructions processed successfully.',
            'recovery_link': action_link
        }), 200
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Password recovery link generation failed for {email}: {err_msg}")
        if "user not found" in err_msg.lower() or "not found" in err_msg.lower():
            return jsonify({
                'status': 'success',
                'message': 'Password recovery instructions processed successfully.'
            }), 200
        return jsonify({
            'status': 'error',
            'error': f'Failed to generate recovery link: {err_msg}'
        }), 500



