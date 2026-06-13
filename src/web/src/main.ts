import "./style.css";
import { ApiError, fetchProductImage, recognize, type GridCell, type MachineDetectionResult } from "./api";
import { CameraError, captureFrame, startCamera, stopCamera } from "./camera";
import { getToken } from "./settings";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("#app root element not found");

app.innerHTML = `
  <h1>Vending Machine Parser</h1>
  <p class="subtitle">Capture a photo of a vending machine and let the backend recognize its product grid.</p>

  <section id="camera-section">
    <h2>Camera</h2>
    <p class="hint">Frame the shot so the whole vending machine fits inside the preview — cropped edges make the grid harder to recognize.</p>
    <video id="preview" autoplay playsinline muted></video>
    <canvas id="capture-canvas"></canvas>
    <div class="row camera-controls">
      <button id="start-camera" type="button">Start camera</button>
      <button id="stop-camera" class="secondary" type="button" disabled>Stop camera</button>
      <button id="capture-send" type="button" disabled>Capture &amp; recognize</button>
    </div>
    <div id="camera-status"></div>

    <p class="divider">— or —</p>

    <div id="upload-dropzone" class="dropzone" tabindex="0" role="button"
         aria-label="Drop an image here or click to choose a file">
      <p>Drag &amp; drop a photo here, or click to choose a file</p>
      <input id="upload-input" type="file" accept="image/*" hidden />
    </div>
    <div id="upload-status"></div>
  </section>

  <section id="results-section" hidden>
    <h2>Recognition result</h2>
    <div id="results"></div>
  </section>

  <footer>Vending Machine Parser — frontend</footer>
`;

// ---------------------------------------------------------------------------
// Element references
// ---------------------------------------------------------------------------

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

const video = byId<HTMLVideoElement>("preview");
const canvas = byId<HTMLCanvasElement>("capture-canvas");
const startCameraBtn = byId<HTMLButtonElement>("start-camera");
const stopCameraBtn = byId<HTMLButtonElement>("stop-camera");
const captureSendBtn = byId<HTMLButtonElement>("capture-send");
const cameraStatus = byId<HTMLDivElement>("camera-status");

const uploadDropzone = byId<HTMLDivElement>("upload-dropzone");
const uploadInput = byId<HTMLInputElement>("upload-input");
const uploadStatus = byId<HTMLDivElement>("upload-status");

const resultsSection = byId<HTMLElement>("results-section");
const resultsContainer = byId<HTMLDivElement>("results");

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

type StatusKind = "info" | "error" | "success" | "";

function showStatus(el: HTMLElement, message: string, kind: StatusKind = "info"): void {
  el.textContent = message;
  el.className = kind ? `status ${kind}` : "muted";
}

function describeUnknownError(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

let cameraRunning = false;

startCameraBtn.addEventListener("click", async () => {
  startCameraBtn.disabled = true;
  showStatus(cameraStatus, "Requesting camera access…", "info");
  try {
    await startCamera(video);
    cameraRunning = true;
    stopCameraBtn.disabled = false;
    captureSendBtn.disabled = false;
    showStatus(cameraStatus, "Camera is live. Frame the vending machine and capture a frame.", "success");
  } catch (err) {
    cameraRunning = false;
    startCameraBtn.disabled = false;
    const message = err instanceof CameraError ? err.message : describeUnknownError(err);
    showStatus(cameraStatus, message, "error");
  }
});

stopCameraBtn.addEventListener("click", () => {
  stopCamera(video);
  cameraRunning = false;
  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  captureSendBtn.disabled = true;
  showStatus(cameraStatus, "Camera stopped.", "info");
});

captureSendBtn.addEventListener("click", async () => {
  if (!cameraRunning) return;

  captureSendBtn.disabled = true;
  showStatus(cameraStatus, "Capturing frame…", "info");

  try {
    const blob = await captureFrame(video, canvas);
    showStatus(cameraStatus, "Sending photo to backend for recognition…", "info");

    const token = getToken();
    if (!token) {
      showStatus(
        cameraStatus,
        "No access token configured — open this page with ?token=<your-token> in the URL.",
        "error",
      );
      return;
    }

    const response = await recognize(token, blob);
    renderResults(response.machines, token);
    showStatus(
      cameraStatus,
      response.machines.length > 0
        ? `Detected ${response.machines.length} vending machine(s).`
        : "No vending machine was detected in the photo. Try moving closer or improving lighting.",
      response.machines.length > 0 ? "success" : "info",
    );
  } catch (err) {
    const message =
      err instanceof ApiError || err instanceof CameraError ? err.message : describeUnknownError(err);
    showStatus(cameraStatus, message, "error");
  } finally {
    if (cameraRunning) captureSendBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

async function recognizeUploadedFile(file: File): Promise<void> {
  if (!file.type.startsWith("image/")) {
    showStatus(uploadStatus, "Please choose an image file.", "error");
    return;
  }

  const token = getToken();
  if (!token) {
    showStatus(
      uploadStatus,
      "No access token configured — open this page with ?token=<your-token> in the URL.",
      "error",
    );
    return;
  }

  uploadDropzone.classList.add("disabled");
  showStatus(uploadStatus, "Sending photo to backend for recognition…", "info");

  try {
    const response = await recognize(token, file, file.name || "upload.jpg");
    renderResults(response.machines, token);
    showStatus(
      uploadStatus,
      response.machines.length > 0
        ? `Detected ${response.machines.length} vending machine(s).`
        : "No vending machine was detected in the photo. Try a clearer or closer shot.",
      response.machines.length > 0 ? "success" : "info",
    );
  } catch (err) {
    const message = err instanceof ApiError ? err.message : describeUnknownError(err);
    showStatus(uploadStatus, message, "error");
  } finally {
    uploadDropzone.classList.remove("disabled");
    uploadInput.value = "";
  }
}

uploadDropzone.addEventListener("click", () => uploadInput.click());
uploadDropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    uploadInput.click();
  }
});

uploadInput.addEventListener("change", () => {
  const file = uploadInput.files?.[0];
  if (file) void recognizeUploadedFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadDropzone.classList.add("dragover");
  });
});

["dragleave", "dragend", "drop"].forEach((eventName) => {
  uploadDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadDropzone.classList.remove("dragover");
  });
});

uploadDropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) void recognizeUploadedFile(file);
});

// Stop the camera when the page is unloaded so the device LED turns off.
window.addEventListener("beforeunload", () => stopCamera());

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

function renderResults(machines: MachineDetectionResult[], token: string): void {
  resultsContainer.innerHTML = "";

  if (machines.length === 0) {
    resultsSection.hidden = false;
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "No vending machine detected in this photo.";
    resultsContainer.appendChild(p);
    return;
  }

  machines.forEach((machine, index) => {
    const block = document.createElement("div");
    block.className = "machine-block";

    const heading = document.createElement("h3");
    heading.textContent = `Machine ${index + 1}: ${machine.machine_info?.name ?? "Unknown"}`;
    block.appendChild(heading);

    const meta = document.createElement("p");
    meta.className = "muted";
    meta.textContent = `bbox: [${machine.bbox.map((v) => v.toFixed(1)).join(", ")}]`;
    block.appendChild(meta);

    if (!machine.grid) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "The shelf window for this machine could not be segmented (grid is unavailable).";
      block.appendChild(note);
    } else {
      const grid = machine.grid;

      if (grid.image) {
        const img = document.createElement("img");
        img.className = "result-image";
        img.alt = `Rectified shelf grid for machine ${index + 1}`;
        img.src = `data:image/jpeg;base64,${grid.image}`;
        block.appendChild(img);
      }

      const dims = document.createElement("p");
      dims.className = "muted";
      dims.textContent = `Grid: ${grid.n_rows} rows × ${grid.n_cols} cols`;
      block.appendChild(dims);

      const tableWrapper = document.createElement("div");
      tableWrapper.className = "table-wrapper";
      tableWrapper.appendChild(buildGridTable(grid.cells, grid.n_rows, grid.n_cols, token));
      block.appendChild(tableWrapper);

      if (grid.markdown) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Show raw markdown table";
        const pre = document.createElement("pre");
        pre.style.whiteSpace = "pre-wrap";
        pre.style.fontSize = "0.78rem";
        pre.textContent = grid.markdown;
        details.appendChild(summary);
        details.appendChild(pre);
        block.appendChild(details);
      }
    }

    resultsContainer.appendChild(block);
  });

  resultsSection.hidden = false;
}

// Cache product reference images by name (lowercased) so repeated cells
// (and repeated recognitions) share a single authenticated fetch + object URL.
const productImageCache = new Map<string, Promise<string | null>>();

function getProductImageUrl(token: string, name: string): Promise<string | null> {
  const key = name.toLowerCase();
  let cached = productImageCache.get(key);
  if (!cached) {
    cached = fetchProductImage(token, name).catch(() => null);
    productImageCache.set(key, cached);
  }
  return cached;
}

function buildGridTable(cells: GridCell[], nRows: number, nCols: number, token: string): HTMLTableElement {
  // Build a lookup from "row:col" -> cell, and mark columns covered by a
  // col_span > 1 so we don't render duplicate <td>s for them.
  const byPosition = new Map<string, GridCell>();
  const spanned = new Set<string>();

  for (const cell of cells) {
    byPosition.set(`${cell.row}:${cell.col}`, cell);
    for (let c = cell.col + 1; c < cell.col + cell.col_span; c++) {
      spanned.add(`${cell.row}:${c}`);
    }
  }

  const table = document.createElement("table");
  table.className = "grid-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th")).textContent = "Row";
  for (let c = 0; c < nCols; c++) {
    headRow.appendChild(document.createElement("th")).textContent = `C${c}`;
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (let r = 0; r < nRows; r++) {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.scope = "row";
    rowHeader.textContent = String(r);
    tr.appendChild(rowHeader);

    for (let c = 0; c < nCols; c++) {
      const key = `${r}:${c}`;
      if (spanned.has(key)) continue;

      const cell = byPosition.get(key);
      const td = document.createElement("td");
      if (cell) {
        if (cell.col_span > 1) td.colSpan = cell.col_span;
        if (cell.product_name) {
          const name = cell.product_name;
          // product_score is a cosine similarity in (-1, 1) — rescale to a 0-100% match.
          const score =
            cell.product_score != null
              ? ` (${Math.round(((cell.product_score + 1) / 2) * 100)}%)`
              : "";

          // Apply the flex layout to a wrapper <div>, not the <td> itself —
          // overriding a table cell's display to "flex" breaks table layout
          // (borders, row heights, column alignment all collapse).
          const wrapper = document.createElement("div");
          wrapper.className = "product-cell";

          const img = document.createElement("img");
          img.className = "cell-product-image";
          img.alt = name;
          img.hidden = true;
          wrapper.appendChild(img);

          const label = document.createElement("span");
          label.textContent = `${name}${score}`;
          wrapper.appendChild(label);

          td.appendChild(wrapper);

          getProductImageUrl(token, name).then((url) => {
            if (url) {
              img.src = url;
              img.hidden = false;
            }
          });
        } else {
          td.textContent = "—";
          td.className = "muted";
        }
      } else {
        td.textContent = "";
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  return table;
}

