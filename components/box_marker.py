"""
box_marker -- a small custom Streamlit component for drag-to-mark-a-box
on an image, with LIVE visual feedback while dragging (a yellow
rubber-band rectangle that follows the cursor).

Built because streamlit-image-coordinates' click_and_drag mode,
confirmed by reading its actual frontend source, only listens for
mousedown/mouseup -- there's no mousemove handling at all, so nothing
is drawn until the drag is released. This component adds an overlay
<canvas> and a mousemove listener to redraw the box on every frame
while dragging, which is the piece that was actually missing.

No new PyPI dependency, no build tooling -- just three static files
(index.html, main.js, style.css) plus Streamlit's own component
protocol bridge (streamlit-component-lib.js, the same file
streamlit-image-coordinates itself ships and uses), served directly
from a local folder via declare_component(path=...).

Returns the same {x1, y1, x2, y2, width, height, unix_time} shape
streamlit-image-coordinates' click_and_drag mode returns, so it's a
drop-in replacement wherever that shape is already being parsed.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

_frontend_dir = (Path(__file__).parent / "box_marker").absolute()
_component_func = components.declare_component("box_marker", path=str(_frontend_dir))


def box_marker(source: np.ndarray, key: str | None = None):
    """source: RGB numpy array (same convention as our other image
    helpers in this project). Returns a dict with x1,y1,x2,y2 in
    DISPLAYED PIXEL space (same as streamlit-image-coordinates), or
    None before any drag has happened."""
    image = Image.fromarray(source)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    src = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return _component_func(src=src, width=image.width, height=image.height, key=key, default=None)
