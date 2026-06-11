// The access token is provided via the `?token=<value>` URL query parameter —
// there is no settings UI and nothing is persisted. Note that
// /products/{name}/image also requires this token, so product images must be
// fetched with `fetch()` + Authorization header (see fetchProductImage), not
// plain <img src>.

export function getToken(): string {
  return new URLSearchParams(window.location.search).get("token") ?? "";
}
