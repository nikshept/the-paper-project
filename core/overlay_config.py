"""
Global placement config for corner markers + QR code, plus a uniform
content-scale factor, applied to an already-finished PDF.

All four marker margins are INDEPENDENT (left_x, right_x, top_y,
bottom_y) -- exactly like Word's page margins, not a single shared
inset. They control ONLY marker/QR placement -- they do NOT affect
where the researcher's content is scaled or positioned; that's
content_scale's job alone (see overlay.py's _content_target_rect).
These two controls were coupled in an earlier version (margins also
defined the area content scaled into) and were deliberately split
apart: adjusting a margin to reposition a marker shouldn't also move
or resize the survey content underneath it.

QR position: horizontally, it's always centered between the left/right
margins -- not freely adjustable, by design. Vertically, the only
choice is "top" or "bottom", and whichever is picked, the QR is
vertically CENTER-ALIGNED with the corner markers on that same edge
(so it visually sits on the same line as the markers, regardless of
the QR being a different size than them) -- also not a free y
coordinate. This trades flexibility for a result that's always
visually coherent with the markers, per an explicit design decision
(a freely-draggable QR could easily end up misaligned with the markers
it needs to sit alongside). The QR is treated as part of the marker
system, not as content, so it stays tied to the margins.

Units: all values in PDF points, PyMuPDF's coordinate system (origin
top-left, y increases downward).
"""
from dataclasses import dataclass, field


@dataclass
class OverlayConfig:
    # --- content scaling (fully independent of the margins below) ---
    content_scale: float = 1.0    # fraction of the full page the original content is
                                   # scaled to, centered on the page's own center. Not
                                   # affected by left_x/right_x/top_y/bottom_y at all.

    # --- marker margins -- position markers/QR ONLY, never content ---
    left_x: float = 28            # pt inset from the left edge
    right_x: float = 28           # pt inset from the right edge
    top_y: float = 28             # pt inset from the top edge
    bottom_y: float = 38          # pt inset from the bottom edge -- kept larger
                                   # by default, matching p1's finding that
                                   # printers need more clearance at the bottom
                                   # edge (paper-transport mechanics)

    marker_size: float = 28       # pt, width/height of each corner marker
    marker_image_paths: dict = field(default_factory=lambda: {
        0: "assets/aruco_markers/aruco_0.png",  # top-left
        1: "assets/aruco_markers/aruco_1.png",  # top-right
        2: "assets/aruco_markers/aruco_2.png",  # bottom-right
        3: "assets/aruco_markers/aruco_3.png",  # bottom-left
    })  # if any of these four files is missing, overlay.py falls back to
        # drawn vector brackets for ALL corners (not a mix of styles) --
        # see overlay.markers_ready()
    bracket_arm_length: float = 20   # pt, only used when marker images aren't available

    # --- QR code -- treated as part of the marker system, tied to margins ---
    qr_position: str = "bottom"   # "top" | "bottom" -- the only two choices, see
                                   # module docstring for why x/y aren't free
    qr_size: float = 32           # pt
    show_tracking_code_text: bool = True
    qr_text_gap: float = 6        # pt, gap between QR and its adjacent text
    qr_text_size: float = 8       # pt


def resolve_qr_position(cfg: OverlayConfig, page_width: float, page_height: float) -> tuple:
    """Returns (qr_x, qr_y) -- the QR's top-left corner. Horizontally
    centered between the left/right margins; vertically center-aligned
    with whichever edge's corner markers were chosen (top or bottom),
    so the QR always lines up with the markers rather than floating
    independently. Entirely marker-margin-driven, same as the corner
    markers themselves -- never touches content_scale or content
    positioning."""
    margin_left = cfg.left_x
    margin_right = page_width - cfg.right_x
    x = (margin_left + margin_right) / 2 - cfg.qr_size / 2

    if cfg.qr_position == "top":
        marker_center_y = cfg.top_y + cfg.marker_size / 2
    elif cfg.qr_position == "bottom":
        marker_center_y = page_height - cfg.bottom_y - cfg.marker_size / 2
    else:
        raise ValueError(f"qr_position must be 'top' or 'bottom', got {cfg.qr_position!r}")

    y = marker_center_y - cfg.qr_size / 2
    return x, y