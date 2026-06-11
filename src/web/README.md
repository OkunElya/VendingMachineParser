# Vending Machine Parser — Web Frontend

A small Vite + TypeScript (vanilla) single-page app that uses your device camera to
photograph a vending machine, sends the photo to the FastAPI backend's `/recognize`
endpoint, and displays the parsed product grid (rectified shelf image + table with
product reference thumbnails).

## Stack

- [Vite](https://vite.dev/) with the `vanilla-ts` template (plain TypeScript, no
  framework, no UI/state-management libraries — just the DOM).
- Plain CSS (`src/style.css`).
- Access token persisted to `localStorage`.

## Architecture: served by the backend (same host & port)

This app is **served by the FastAPI/uvicorn process itself** — `api.py` mounts
`web/dist` as static files at `/` once it has been built. That means the frontend
and the API share the same origin (e.g. `http://localhost:8000`), so:

- all requests use **relative paths** (`/recognize`, `/products/{name}/image`) —
  there is no "backend base URL" setting, and
- no CORS configuration is needed.

## Building & running

From this directory, build the production bundle:

```sh
npm install
npm run build
```

This produces `web/dist/`. Then, from the **project root**, start the backend —
it will detect `web/dist` and serve the app at the same address as the API:

```sh
python3 api.py
```

Open `http://localhost:8000/` in a browser that supports `getUserMedia` (any
modern desktop or mobile browser).

For frontend-only iteration with hot reload, `npm run dev` also works (Vite serves
on `http://localhost:5173`); just make sure the backend is also running on
`:8000` — note that in this mode the origins differ, so you would need to add CORS
middleware back to `api.py` for dev-server requests to succeed. The supported,
CORS-free path is build + serve through `api.py` as described above.

## Using the app

1. Open `http://localhost:8000/?token=<your-token>` (see **Access token** above).
2. In the **Camera** section, click **Start camera** and grant camera permission
   when prompted.
3. Frame the vending machine in the live preview and click **Capture & recognize**.
   The captured frame is sent as `multipart/form-data` (field name `file`, JPEG)
   to `POST /recognize`.
4. The **Recognition result** section shows, for each detected machine:
   - the rectified shelf image returned by the backend (with grid lines/labels
     drawn on it, decoded from the base64 `image` field), and
   - an HTML table built from the structured `cells` data — each recognized cell
     shows the product name, similarity score, **and a reference thumbnail**
     fetched from `GET /products/{name}/image` (cached per product name so it's
     only fetched once), plus a collapsible view of the raw `markdown` table.

## Notes / caveats

- **Camera requires a secure context.** Browsers only allow
  `navigator.mediaDevices.getUserMedia` on `https://` origins or on `localhost`
  (`http://localhost:8000` works out of the box). If you deploy this app or access
  it from another device on your network via plain HTTP, the browser will block
  camera access — serve it over HTTPS in that case.
- The camera stream is stopped automatically when you click **Stop camera** or
  leave/reload the page, so the camera indicator light turns off.
