import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

export default {
  // Compile to standard components (no customElement output). Shoelace custom
  // elements are consumed as plain tags, not authored here.
  preprocess: vitePreprocess(),
};
