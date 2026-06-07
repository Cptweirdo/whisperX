<script lang="ts">
  import { router } from "./lib/router.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import Topbar from "./components/Topbar.svelte";
  import Dashboard from "./routes/Dashboard.svelte";
  import Transcript from "./routes/Transcript.svelte";
  import Settings from "./routes/Settings.svelte";
  import Onboarding from "./routes/Onboarding.svelte";

  const route = $derived(router.current);
  // Onboarding is a standalone full-bleed screen (no app chrome).
  const chrome = $derived(route.name !== "onboarding");
</script>

{#if chrome}
  <div class="app">
    <Sidebar />
    <div class="app__main">
      <Topbar />
      <main class="content">
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
      </main>
    </div>
  </div>
{:else}
  <Onboarding />
{/if}
