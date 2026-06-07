// Toast notifications via Shoelace's <sl-alert>.toast(), ported from base.html.

type Variant = "primary" | "success" | "neutral" | "warning" | "danger";

const ICONS: Record<Variant, string> = {
  primary: "info-circle",
  success: "check2-circle",
  neutral: "gear",
  warning: "exclamation-triangle",
  danger: "exclamation-octagon",
};

function escape(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!),
  );
}

/** Show a transient toast. Returns the alert element. */
export function notify(message: string, variant: Variant = "primary", duration = 3500): HTMLElement {
  const alert = Object.assign(document.createElement("sl-alert"), {
    variant,
    closable: true,
    duration,
    innerHTML: `<sl-icon slot="icon" name="${ICONS[variant]}"></sl-icon>${escape(message)}`,
  }) as any;
  document.body.append(alert);
  // toast() upgrades + shows once the custom element is defined.
  customElements.whenDefined("sl-alert").then(() => alert.toast());
  return alert;
}
