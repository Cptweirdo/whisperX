<script lang="ts">
  import { onMount } from "svelte";
  import { sessions } from "../lib/stores/sessions.svelte";
  import { ui } from "../lib/stores/ui.svelte";
  import FeatureCard from "../components/dashboard/FeatureCard.svelte";
  import LibrarySummary from "../components/dashboard/LibrarySummary.svelte";
  import QueueTable from "../components/dashboard/QueueTable.svelte";
  import NewRecordingModal from "../components/dashboard/NewRecordingModal.svelte";
  import RenameModal from "../components/dashboard/RenameModal.svelte";
  import DeleteModal from "../components/dashboard/DeleteModal.svelte";

  let newModal = $state<NewRecordingModal | null>(null);
  let renameModal = $state<RenameModal | null>(null);
  let deleteModal = $state<DeleteModal | null>(null);

  onMount(async () => {
    if (!sessions.loaded) await sessions.load();
    // Honour a New Recording intent from another page, or ?new=1 deep link.
    if (ui.consumeNewRecording() || new URLSearchParams(location.search).has("new")) {
      newModal?.open();
      history.replaceState(null, "", "/");
    }
  });

  // React to a New Recording request raised while already on the dashboard.
  $effect(() => {
    if (ui.newRecordingRequested) {
      ui.consumeNewRecording();
      newModal?.open();
    }
  });

  function onRenamed(id: string, newName: string) {
    const row = sessions.list.find((c) => c.id === id);
    if (row) row.name = newName;
  }
</script>

<div class="dash">
  <h1 class="dash__h">Recent Recordings</h1>
  <p class="dash__sub">Manage and transcribe your captured audio.</p>

  <div class="dash__grid">
    {#if sessions.featured}
      <FeatureCard
        card={sessions.featured}
        onrename={(id, name) => renameModal?.open(id, name)}
        ondelete={(id, name) => deleteModal?.open(id, name)}
      />
    {:else}
      <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions, a11y_no_noninteractive_element_interactions -->
      <article class="feature feature--empty" onclick={() => newModal?.open()}>
        <div class="dropzone__icon"><sl-icon name="file-earmark-arrow-up"></sl-icon></div>
        <h2 class="feature__title" style="margin-top:0">No recordings yet</h2>
        <p class="feature__desc" style="max-width:420px">
          Upload an audio file to generate your first manuscript draft.
        </p>
        <div style="margin-top:24px"><sl-button variant="primary">New Recording</sl-button></div>
      </article>
    {/if}

    <LibrarySummary summary={sessions.summary} />
  </div>

  <QueueTable
    onrename={(id, name) => renameModal?.open(id, name)}
    ondelete={(id, name) => deleteModal?.open(id, name)}
  />
</div>

<NewRecordingModal bind:this={newModal} />
<RenameModal bind:this={renameModal} onsaved={onRenamed} />
<DeleteModal bind:this={deleteModal} onconfirm={(id) => sessions.remove(id)} />
