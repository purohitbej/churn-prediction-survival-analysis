"""
wsgi.py
-------
Entry point for running the Flask app with gunicorn.

Production usage
----------------
    gunicorn wsgi:app

Development usage
-----------------
    python wsgi.py
"""

from api.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
