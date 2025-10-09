# ==================================
# Jules AI Agent Configuration
# For the ESG AutoScorer Project
# ==================================

# Project Context
You are an expert Python developer working on a web application named "ESG AutoScorer".
The project's goal is to automatically score ESG sustainability reports by leveraging Google's Generative AI.

# --- Tech Stack ---
- The backend is a single-file application built with FastAPI and served by Uvicorn.
- The core AI logic uses the `google-generativeai` library to interact with Google's AI models.
- The application handles PDF file uploads (`pypdf`, `python-multipart`) for analysis.
- The frontend is a single `index.html` file that uses JavaScript and AJAX to communicate with the backend API without page reloads.
- Environment variables are managed using `.env` files.

# Instructions
# --- General ---
- All Python code must be compatible with Python 3.9.
- Use modern f-strings for string formatting.
- Add clear docstrings and type hints to all major functions, especially those in `main.py`.

# --- Project Structure & Logic ---
- The entire backend logic, including API endpoints and file processing, is contained within `main.py`.
- The frontend interface is in `index.html`.
- All tests, written using `pytest`, must be located in the `tests/` directory.

# --- Coding Style ---
- Keep functions concise and focused on a single task.
- Use the `logging` module for server-side logging instead of `print()`. This is important for a web server.
- Ensure API responses are in a clear JSON format that the frontend JavaScript can easily handle.

# Ignored Files
# Telling Jules to ignore these files to save time and prevent errors.
- `__pycache__/`
- `venv/`
- `.venv/`
- `*.pyc`
- `.env`
- `.gitignore`
- `pyvenv.cfg`
- `.DS_Store`
- `NotoSansTC-Regular.ttf` # Ignoring the font file