// Opaque participant token: generated entirely client-side via the
// browser's cryptographically-secure UUID generator, with NO link to any
// real identity (no name/email/IP is ever collected anywhere in this
// app -- see study/protocol.md's Ethics section). Persisted in
// localStorage only so a browser refresh mid-task doesn't orphan an
// in-progress response; nothing about the token identifies the device
// beyond "the same browser session used this token before."

const STORAGE_KEY = "blind_study_token_v1";

export function getOrCreateToken() {
  let token = window.localStorage.getItem(STORAGE_KEY);
  if (!token) {
    token = crypto.randomUUID();
    window.localStorage.setItem(STORAGE_KEY, token);
  }
  return token;
}

export function clearToken() {
  window.localStorage.removeItem(STORAGE_KEY);
}
