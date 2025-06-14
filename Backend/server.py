from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Backend.main import app as main_app

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the main app as API
app.mount("/api", main_app)

# Mount the static files (frontend)
app.mount("/", StaticFiles(directory="Frontend", html=True), name="static")

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info") 