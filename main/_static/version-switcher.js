(function () {
  function detectBase() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    if (parts.length > 0 && parts[0].toLowerCase() === "playnano") return "/playNano";
    return "";  // local server or custom domain at root
  }
  const BASE = detectBase();
  const VERSIONS_URL = `${BASE}/versions.json?v=${Date.now()}`; // avoid cache during testing

  const mount = document.getElementById("version-switcher-placeholder");
  if (!mount) {
    console.warn("[version-switcher] Placeholder #version-switcher-placeholder not found.");
    return;
  }

  function normalisePath(p) {
    try { const u = new URL(p, window.location.origin); return u.pathname.endsWith("/") ? u.pathname : (u.pathname + "/"); }
    catch { return "/"; }
  }

  function currentVersionFromUrl(versions) {
    const path = normalisePath(window.location.pathname);
    return versions.find(v => path.startsWith(normalisePath(v.url))) || null;
  }

  function buildSelect(versions, current) {
    const wrapper = document.createElement("div");

    const label = document.createElement("label");
    label.setAttribute("for", "version-switcher-select");
    label.style.display = "none";
    label.textContent = "Select documentation version";
    wrapper.appendChild(label);

    const select = document.createElement("select");
    select.id = "version-switcher-select";
    select.className = "version-switcher__select";

    versions.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v.url;
      opt.textContent = v.title || v.name;
      if (current && current.name === v.name) opt.selected = true;
      select.appendChild(opt);
    });

    select.addEventListener("change", () => {
      const targetBase = select.value;
      const cur = currentVersionFromUrl(versions);
      const currentPath = window.location.pathname;
      let remainder = "";
      if (cur) {
        const curBase = normalisePath(cur.url);
        if (currentPath.startsWith(curBase)) remainder = currentPath.slice(curBase.length);
      }
      const newUrl = new URL(
        targetBase.replace(/\/?$/, "/") + remainder.replace(/^\//, ""),
        window.location.origin
      );
      newUrl.search = window.location.search;
      newUrl.hash = window.location.hash;
      window.location.assign(newUrl.toString());
    });

    wrapper.appendChild(select);
    return wrapper;
  }

  fetch(VERSIONS_URL, { cache: "no-store" })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${VERSIONS_URL}`); return r.json(); })
    .then(versions => {
      if (!Array.isArray(versions) || versions.length === 0) {
        console.warn("[version-switcher] versions.json is empty or invalid:", versions);
        return;
      }
      const current = currentVersionFromUrl(versions);
      mount.appendChild(buildSelect(versions, current));
    })
    .catch(err => { console.error("[version-switcher] Failed to load versions.json:", err); });
})();