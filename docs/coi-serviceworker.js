/*! coi-serviceworker v0.1.7 - Guido Zuidhof and contributors, licensed under MIT */
/* Adds COOP/COEP headers so SharedArrayBuffer (and multi-threaded WASM) works. */

let coepCredentialless = false;
if (typeof window === 'undefined') {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener("message", (ev) => {
    if (!ev.data) return;
    if (ev.data.type === "deregister") {
      self.registration.unregister().then(() => {
        return self.clients.matchAll();
      }).then((clients) => {
        clients.forEach((client) => client.navigate(client.url));
      });
    } else if (ev.data.type === "coepCredentialless") {
      coepCredentialless = ev.data.value;
    }
  });

  self.addEventListener("fetch", function (event) {
    const r = event.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;

    const request = (coepCredentialless && r.mode === "no-cors")
      ? new Request(r, { credentials: "omit" })
      : r;

    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response;

          const newHeaders = new Headers(response.headers);
          newHeaders.set("Cross-Origin-Embedder-Policy", coepCredentialless ? "credentialless" : "require-corp");
          if (!coepCredentialless) {
            newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");
          }
          newHeaders.set("Cross-Origin-Resource-Policy", "cross-origin");

          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: newHeaders,
          });
        })
        .catch((e) => console.error(e))
    );
  });
} else {
  // Capture current script src SYNCHRONOUSLY (before it becomes null)
  const currentScriptSrc = (document.currentScript && document.currentScript.src)
    ? document.currentScript.src
    : new URL('coi-serviceworker.js', window.location.href).href;

  (() => {
    const reloadedBySelf = window.sessionStorage.getItem("coiReloadedBySelf");
    window.sessionStorage.removeItem("coiReloadedBySelf");
    const coepDegrading = (reloadedBySelf == "coepdegrade");

    const coi = {
      shouldRegister: () => !reloadedBySelf,
      shouldDeregister: () => false,
      coepCredentialless: () => true,
      coepDegrade: () => true,
      doReload: () => window.location.reload(),
      quiet: false,
      ...window.coi
    };

    const n = navigator;
    const controlling = n.serviceWorker && n.serviceWorker.controller;

    if (controlling && !window.crossOriginIsolated) {
      window.sessionStorage.setItem("coiCoepHasFailed", "true");
    }
    const coepHasFailed = window.sessionStorage.getItem("coiCoepHasFailed");

    if (controlling) {
      const reloadToDegrade = () => {
        if (coepDegrading) {
          window.sessionStorage.setItem("coiReloadedBySelf", "coepdegrade");
        } else {
          window.sessionStorage.setItem("coiReloadedBySelf", "true");
        }
        coi.doReload();
      };

      if (coi.coepDegrade() && !(coepDegrading || window.crossOriginIsolated)) {
        reloadToDegrade();
        return;
      }

      if (!coi.coepCredentialless() && coepHasFailed && window.crossOriginIsolated) {
        reloadToDegrade();
        return;
      }
    }

    if (coi.shouldDeregister()) {
      n.serviceWorker.controller.postMessage({ type: "deregister" });
    }

    if (!controlling && coi.shouldRegister()) {
      if (n.serviceWorker) {
        window.addEventListener("load", () => {
          n.serviceWorker.register(currentScriptSrc)
            .then(() => {
              if (n.serviceWorker.controller) {
                window.sessionStorage.setItem("coiReloadedBySelf", "true");
                coi.doReload();
              }
            })
            .catch(console.error);
        });
      }
    }
  })();
}