/* Public repository metadata only; no token is needed or shipped to browsers. */
(async () => {
  const link = document.getElementById("github-link");
  const counter = document.getElementById("github-stars");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 6000);
  try {
    const response = await fetch("https://api.github.com/repos/Bring-AI/jev-rl", {
      headers: { Accept: "application/vnd.github+json" },
      signal: controller.signal,
      credentials: "omit",
    });
    if (!response.ok) throw new Error("Repository metadata unavailable");
    const { stargazers_count: stars } = await response.json();
    if (!Number.isSafeInteger(stars) || stars < 0) throw new Error("Invalid star count");
    counter.textContent = new Intl.NumberFormat("en-US").format(stars);
    link.title = `${stars.toLocaleString("en-US")} GitHub stars`;
    link.setAttribute("aria-label", `JevRL on GitHub, ${stars} stars`);
  } catch {
    link.title = "View JevRL on GitHub · Star count temporarily unavailable";
  } finally {
    clearTimeout(timeout);
  }
})();
