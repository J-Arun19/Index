import os
import sys

# Ensure the parent directory is in sys.path so we can import 'backend'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.app import create_app

app = create_app()

if __name__ == '__main__':
    print("Starting SatyaDrishti Flask Backend...")
    # Run server locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
