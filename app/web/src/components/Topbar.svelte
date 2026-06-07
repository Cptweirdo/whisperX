<script lang="ts">
  import { router } from "../lib/router.svelte";

  const TABS: Record<string, string> = {
    dashboard: "Dashboard",
    transcript: "Transcript",
    settings: "Settings",
  };
  const tab = $derived(TABS[router.current.name] ?? "Dashboard");
</script>

<header class="topbar">
  <div class="topbar__tabs">
    <span class="topbar__tab topbar__tab--active">{tab}</span>
  </div>
  <div style="flex:1"></div>
  <div class="topbar__right"></div>
</header>
