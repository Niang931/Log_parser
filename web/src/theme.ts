// The colour theme: "auto" follows the OS. Kept per browser; applied before first paint by main.tsx.
export type Theme = "auto" | "light" | "dark";

export function getTheme(): Theme {
  try { return (localStorage.getItem("theme") as Theme | null) ?? "auto"; } catch { return "auto"; }
}

export function setTheme(theme: Theme) {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("theme", theme); } catch { /* private window */ }
}
