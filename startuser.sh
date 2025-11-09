#!/bin/bash
# Navigate to the User directory
cd User || exit

# Activate the virtual environment
source venv/bin/activate

# Run the Python app
python app.py
