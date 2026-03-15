import sys
import traceback

import uvicorn

if __name__ == "__main__":
    try:
        from intelligent_search.config import get_settings

        settings = get_settings()
        uvicorn.run(
            "intelligent_search.main:app",
            host=settings.host,
            port=settings.port,
            reload=False,
        )
    except Exception:
        print("FATAL: failed to start — see traceback below", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
