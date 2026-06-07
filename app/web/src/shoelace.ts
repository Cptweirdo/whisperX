// Shoelace, cherry-picked to the components the UI uses. Web components
// self-register on import; load order is not critical, but this module must be
// imported before the app mounts so custom elements upgrade in place.
import { setBasePath } from "@shoelace-style/shoelace/dist/utilities/base-path.js";

// sl-icon fetches Bootstrap Icons by name at runtime from <basePath>/assets/icons/.
// vite.config.ts serves these under /shoelace in dev and copies them into the
// build output, so BASE_URL + "shoelace" resolves in both dev and prod.
setBasePath(import.meta.env.BASE_URL + "shoelace");

import "@shoelace-style/shoelace/dist/components/icon/icon.js";
import "@shoelace-style/shoelace/dist/components/button/button.js";
import "@shoelace-style/shoelace/dist/components/input/input.js";
import "@shoelace-style/shoelace/dist/components/select/select.js";
import "@shoelace-style/shoelace/dist/components/option/option.js";
import "@shoelace-style/shoelace/dist/components/dialog/dialog.js";
import "@shoelace-style/shoelace/dist/components/alert/alert.js";
import "@shoelace-style/shoelace/dist/components/spinner/spinner.js";
import "@shoelace-style/shoelace/dist/components/range/range.js";
