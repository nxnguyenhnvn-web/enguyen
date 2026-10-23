"""Entry point used by the enguyen.streamlit.app deployment (its main file is smoke_test.py); runs app.py.
The headless data/forecast check lives in cli_test.py."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), run_name="__main__")
