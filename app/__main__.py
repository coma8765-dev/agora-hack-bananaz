import os

import uvicorn

uvicorn.run(
    "app.api:app",
    host=os.getenv("HOST") or "0.0.0.0",
    port=int(os.getenv("PORT") or 8000),
    reload=bool(os.getenv("DEBUG", True)),
    debug=False,
)
