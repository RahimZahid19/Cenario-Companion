from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import core, analytics, util
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(core.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(util.router, prefix="/api")

def main():
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)


if __name__ == "__main__":
    main()
