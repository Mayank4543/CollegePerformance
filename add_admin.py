from flask import request

@app.before_request
def check_jwt():
    if request.endpoint in ["admin_dashboard"]:  # Secure admin route
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
