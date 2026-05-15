from __future__ import annotations

import os

from app import app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
        return

    serve(app, host=host, port=port)


if __name__ == "__main__":
    main()
