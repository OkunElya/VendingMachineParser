// Thin client for the FastAPI vending-machine-parser backend.

export interface MachineInfo {
  name: string;
  max_rows: number;
  max_cols: number;
  [key: string]: unknown;
}

export interface GridCell {
  row: number;
  col: number;
  col_span: number;
  product_name: string | null;
  product_score: number | null;
}

export interface GridResult {
  n_rows: number;
  n_cols: number;
  markdown: string;
  cells: GridCell[];
  image: string | null; // base64 JPEG, no data: prefix
}

export interface MachineDetectionResult {
  machine_info: MachineInfo;
  bbox: [number, number, number, number];
  grid: GridResult | null;
}

export interface RecognizeResponse {
  machines: MachineDetectionResult[];
}

/** Error raised for any non-2xx response from the backend. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const data = await res.clone().json();
    if (data && typeof data === "object" && "detail" in data) {
      const detail = (data as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      return JSON.stringify(detail);
    }
  } catch {
    // not JSON — fall through to text
  }
  try {
    const text = await res.text();
    if (text) return text;
  } catch {
    // ignore
  }
  return res.statusText || `HTTP ${res.status}`;
}

async function assertOk(res: Response): Promise<void> {
  if (res.ok) return;

  if (res.status === 401) {
    throw new ApiError(401, "Invalid access token — check the ?token=<value> query parameter in the URL.");
  }

  const detail = await readErrorDetail(res);
  throw new ApiError(res.status, `Request failed (${res.status}): ${detail}`);
}

/**
 * POST an image blob to /recognize and return the parsed grid result(s).
 *
 * The frontend is served by the same FastAPI/uvicorn process as the API
 * (same host and port), so a relative path is enough — no base URL needed.
 */
export async function recognize(
  token: string,
  imageBlob: Blob,
  filename = "capture.jpg",
): Promise<RecognizeResponse> {
  const formData = new FormData();
  formData.append("file", imageBlob, filename);

  let res: Response;
  try {
    res = await fetch("/recognize", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });
  } catch (err) {
    throw new ApiError(0, `Could not reach the backend. Is it running? (${describeNetworkError(err)})`);
  }

  await assertOk(res);
  return (await res.json()) as RecognizeResponse;
}

/**
 * Fetch a reference product image as an object URL (caller should revoke it when done).
 *
 * /products/{name}/image requires the bearer token, so the image must be
 * fetched via `fetch()` with an Authorization header — a plain <img src="...">
 * cannot send custom headers — and rendered through an object URL.
 */
export async function fetchProductImage(token: string, name: string): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`/products/${encodeURIComponent(name)}/image`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
  } catch (err) {
    throw new ApiError(0, `Could not reach the backend. Is it running? (${describeNetworkError(err)})`);
  }

  if (res.status === 404) {
    throw new ApiError(404, `No reference image found for product "${name}".`);
  }
  await assertOk(res);

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

function describeNetworkError(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
