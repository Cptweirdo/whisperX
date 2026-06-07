import "./shoelace";
import "./vendor.css"; // Shoelace light theme + self-hosted fonts (Literata / JetBrains Mono)
import "./app.css"; // Manuscript design system (migrated from static/manuscript.css)
import { mount } from "svelte";
import App from "./App.svelte";

const app = mount(App, { target: document.getElementById("app")! });

export default app;
