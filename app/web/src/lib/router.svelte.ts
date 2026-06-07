// Tiny history-API router. Five routes; depends on the Flask catch-all serving
// index.html for any non-API path so deep links / reloads work.

export type RouteName = "dashboard" | "transcript" | "settings" | "onboarding" | "notfound";

export interface Route {
  name: RouteName;
  params: Record<string, string>;
  path: string;
}

function match(path: string): Route {
  if (path === "/" || path === "") return { name: "dashboard", params: {}, path };
  if (path === "/onboarding") return { name: "onboarding", params: {}, path };
  if (path === "/settings") return { name: "settings", params: {}, path };
  const m = /^\/sessions\/([^/]+)\/?$/.exec(path);
  if (m) return { name: "transcript", params: { id: m[1] }, path };
  return { name: "notfound", params: {}, path };
}

class Router {
  current = $state<Route>(match(location.pathname));

  constructor() {
    addEventListener("popstate", () => {
      this.current = match(location.pathname);
    });
  }

  navigate(path: string, { replace = false } = {}) {
    if (path === this.current.path) return;
    if (replace) history.replaceState({}, "", path);
    else history.pushState({}, "", path);
    this.current = match(path);
  }
}

export const router = new Router();

/** Use as an action on <a use:link href="/x"> to navigate client-side. */
export function link(node: HTMLAnchorElement) {
  const onClick = (e: MouseEvent) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const href = node.getAttribute("href");
    if (!href || href.startsWith("http") || node.target) return;
    e.preventDefault();
    router.navigate(href);
  };
  node.addEventListener("click", onClick);
  return { destroy: () => node.removeEventListener("click", onClick) };
}
