#!/bin/bash
python -m PyInstaller --onefile --paths .venv\Lib\site-packages src/main.py
