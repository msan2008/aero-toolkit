"""Registers the Wing Lab game frontend (app/wing_lab_game/index.html) as a
Streamlit custom component.

declare_component() must be called from an importable module (not from a page
script, which Streamlit executes without a module), so it lives here.
"""

from pathlib import Path

import streamlit.components.v1 as components

game_component = components.declare_component(
    "wing_lab_game", path=str(Path(__file__).parent / "wing_lab_game")
)
