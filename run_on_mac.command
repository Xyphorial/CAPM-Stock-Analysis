#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 capm_project.py
echo
echo "CAPM project finished. Open the outputs folder to see the results."
read -r -p "Press Enter to close this window..."

