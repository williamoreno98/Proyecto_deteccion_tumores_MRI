#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001 --host 0.0.0.0
