#!/bin/bash
# Start the FastAPI backend on port 8000
uvicorn backend.main:app --host localhost --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start the Streamlit frontend on port 5000
streamlit run app.py --server.port 5000 --server.address 0.0.0.0

# If streamlit exits, kill backend
kill $BACKEND_PID 2>/dev/null
