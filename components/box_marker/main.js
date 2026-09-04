// The `Streamlit` object exists because index.html includes
// streamlit-component-lib.js (same file streamlit_image_coordinates
// itself uses -- this is Streamlit's own generic protocol bridge, not
// specific to any one component).

function sendValue(value) {
  Streamlit.setComponentValue(value);
}

let dragging = false;
let startX = 0;
let startY = 0;

function clearOverlay(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawBox(canvas, x0, y0, x1, y1) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#ffdc00";
  ctx.lineWidth = 2;
  ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
}

function mouseMoveListener(moveEvent) {
  const img = document.getElementById("image");
  const canvas = document.getElementById("overlay");
  const rect = img.getBoundingClientRect();
  const curX = moveEvent.clientX - rect.left;
  const curY = moveEvent.clientY - rect.top;
  drawBox(canvas, startX, startY, curX, curY);
}

function mouseDownListener(downEvent) {
  const img = document.getElementById("image");
  const canvas = document.getElementById("overlay");
  const rect = img.getBoundingClientRect();
  startX = downEvent.clientX - rect.left;
  startY = downEvent.clientY - rect.top;
  dragging = true;

  window.addEventListener("mousemove", mouseMoveListener);

  window.addEventListener("mouseup", (upEvent) => {
    window.removeEventListener("mousemove", mouseMoveListener);
    if (!dragging) return;
    dragging = false;

    const endX = upEvent.clientX - rect.left;
    const endY = upEvent.clientY - rect.top;
    const unixTime = Date.now();

    sendValue({
      x1: startX, y1: startY, x2: endX, y2: endY,
      width: img.width, height: img.height, unix_time: unixTime,
    });
    // Leave the final box drawn on the overlay as visible confirmation
    // until Python's next render redraws the base image (with the box
    // baked in green/yellow) -- avoids a flash of "nothing there"
    // between mouseup and the rerun completing.
    drawBox(canvas, startX, startY, endX, endY);
  }, { once: true });
}

function onRender(event) {
  const { src, width, height } = event.detail.args;
  const img = document.getElementById("image");
  const canvas = document.getElementById("overlay");

  if (img.src !== src) {
    img.src = src;
  }

  function sizeToImage() {
    if (width) img.width = width;
    if (height) img.height = height;
    if (!width && !height) {
      img.width = img.naturalWidth;
      img.height = img.naturalHeight;
    }
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.style.width = img.width + "px";
    canvas.style.height = img.height + "px";
    clearOverlay(canvas);
    Streamlit.setFrameHeight(img.height);
  }

  img.onload = sizeToImage;
  if (img.complete && img.naturalWidth) {
    sizeToImage();
  }

  img.onmousedown = mouseDownListener;
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
