#!/bin/bash
# Navigate to the Admin directory
cd Admin || exit

# Activate the virtual environment
source venv/bin/activate

# Run the Python app
python app.py
