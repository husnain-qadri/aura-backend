from flask import Flask
from flask_cors import CORS

from app.config import MAX_UPLOAD_SIZE_MB, FLASK_PORT, FLASK_DEBUG


def create_app() -> Flask:
    application = Flask(__name__)
    application.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    CORS(application)

    from app.routes import (
        parse, explain, query, compare, citation,
        references, highlights, reading_path, checkpoints,
    )
    application.register_blueprint(parse.bp)
    application.register_blueprint(explain.bp)
    application.register_blueprint(query.bp)
    application.register_blueprint(compare.bp)
    application.register_blueprint(citation.bp)
    application.register_blueprint(references.bp)
    application.register_blueprint(highlights.bp)
    application.register_blueprint(reading_path.bp)
    application.register_blueprint(checkpoints.bp)

    @application.route('/api/health')
    def health():
        return {'status': 'ok'}

    return application
