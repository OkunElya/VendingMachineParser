// Wraps getUserMedia camera access and still-frame capture.

export class CameraError extends Error {}

let activeStream: MediaStream | null = null;

/**
 * Request camera access and attach the stream to the given <video> element.
 * Returns the active MediaStream so the caller can stop it later.
 */
export async function startCamera(video: HTMLVideoElement): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new CameraError(
      "Camera access is not available in this browser/context. " +
        "Use a recent browser over HTTPS or localhost.",
    );
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
  } catch (err) {
    throw new CameraError(describeGetUserMediaError(err));
  }

  stopCamera();
  activeStream = stream;
  video.srcObject = stream;
  await video.play().catch(() => {
    // Autoplay can reject before user interaction on some browsers; ignore —
    // the video will still start once the stream is attached and visible.
  });
  return stream;
}

/** Stop any currently active camera stream and detach it. */
export function stopCamera(video?: HTMLVideoElement): void {
  if (activeStream) {
    for (const track of activeStream.getTracks()) {
      track.stop();
    }
    activeStream = null;
  }
  if (video) {
    video.srcObject = null;
  }
}

/**
 * Capture the current video frame as a JPEG Blob.
 */
export function captureFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Blob> {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return Promise.reject(new CameraError("Camera is not ready yet — wait for the preview to start."));
  }

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return Promise.reject(new CameraError("Could not access canvas 2D context to capture the frame."));
  }
  ctx.drawImage(video, 0, 0, width, height);

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new CameraError("Failed to encode the captured frame as JPEG."));
      },
      "image/jpeg",
      0.92,
    );
  });
}

function describeGetUserMediaError(err: unknown): string {
  if (err instanceof DOMException) {
    switch (err.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return "Camera permission was denied. Allow camera access in your browser settings and try again.";
      case "NotFoundError":
      case "DevicesNotFoundError":
        return "No camera device was found on this system.";
      case "NotReadableError":
      case "TrackStartError":
        return "The camera is already in use by another application.";
      case "OverconstrainedError":
        return "No camera satisfies the requested constraints.";
      case "SecurityError":
        return "Camera access requires a secure context (HTTPS or localhost).";
      default:
        return `Camera access failed: ${err.message || err.name}`;
    }
  }
  if (err instanceof Error) return err.message;
  return String(err);
}
