"""ImportReady AI - consumer Streamlit entrypoint.

Launch (from the repository root):

    python -m streamlit run app.py

The application then serves at http://localhost:8501 by default. This file is a
thin wrapper so the launch command stays obvious; the UI itself lives in
``src/ui`` and all compliance logic stays in the accepted engine packages.
"""

from __future__ import annotations

from src.ui.app import main

main()
