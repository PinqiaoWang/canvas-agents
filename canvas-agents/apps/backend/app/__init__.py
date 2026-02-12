from flask import Flask
from dotenv import load_dotenv
import os

def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")

    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
