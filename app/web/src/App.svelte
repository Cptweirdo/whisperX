<script lang="ts">
  import { onMount } from "svelte";
  import { router } from "./lib/router.svelte";
  import { models } from "./lib/stores/models.svelte";
  import { backup } from "./lib/stores/backup.svelte";
  import { settings } from "./lib/stores/settings.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import Topbar from "./components/Topbar.svelte";
  import Dashboard from "./routes/Dashboard.svelte";
  import Transcript from "./routes/Transcript.svelte";
  import Settings from "./routes/Settings.svelte";
  import Onboarding from "./routes/Onboarding.svelte";

  const route = $derived(router.current);
  // Onboarding is a standalone full-bleed screen (no app chrome).
  const chrome = $derived(route.name !== "onboarding");

  onMount(async () => {
    // Persistent live streams (open once, survive client-side navigation).
    models.start();
    backup.start();
    try {
      await settings.load(); // seeds models status + the onboarded flag
      if (!settings.onboarded && router.current.name !== "onboarding") {
        router.navigate("/onboarding", { replace: true });
      }
    } catch {
      await models.load().catch(() => {});
    }
  });
</script>

{#if chrome}
  <div class="screen">
    <Sidebar />
    <div class="main">
      <Topbar />
      <div class="content">
        {#if route.name === "dashboard"}
          <Dashboard />
        {:else if route.name === "transcript"}
          <Transcript id={route.params.id} />
        {:else if route.name === "settings"}
          <Settings />
        {:else}
          <div class="empty">
            <h2>Not found</h2>
            <p>That page doesn't exist.</p>
          </div>
        {/if}
      </div>
    </div>
  </div>
{:else}
  <Onboarding />
{/if}
