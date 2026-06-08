<script lang="ts">
  import { session } from "../lib/stores/session.svelte";
  import { router, link } from "../lib/router.svelte";
  import { urls } from "../lib/api";
  import AudioPlayer from "../components/transcript/AudioPlayer.svelte";
  import TranscriptBody from "../components/transcript/TranscriptBody.svelte";
  import TranslateChip from "../components/transcript/TranslateChip.svelte";
  import AddTranslationModal from "../components/transcript/AddTranslationModal.svelte";
  import SpeakerRenameModal from "../components/transcript/SpeakerRenameModal.svelte";
  import RenameModal from "../components/dashboard/RenameModal.svelte";

  let { id }: { id: string } = $props();

  let player = $state<AudioPlayer | null>(null);
  let currentTime = $state(0);
  let addModal = $state<AddTranslationModal | null>(null);
  let speakerModal = $state<SpeakerRenameModal | null>(null);
  let renameModal = $state<RenameModal | null>(null);
  let loading = $state(true);

  $effect(() => {
    const sid = id;
    loading = true;
    session.load(sid).then(() => {
      loading = false;
      if (session.row && session.row.status !== "done") router.navigate("/");
    });
  });

  const row = $derived(session.row);
</script>

{#if loading || !row}
  <div style="padding:60px;text-align:center"><sl-spinner style="font-size:2rem"></sl-spinner></div>
{:else}
  <div class="tr">
    <div class="tr__scroll">
      <a class="tr__back" href="/" use:link><sl-icon name="arrow-left"></sl-icon> ALL RECORDINGS</a>

      <div class="tr__hero-meta">
        <TranslateChip onadd={() => addModal?.open()} />
        <span class="chip">{row.diarized ? "Diarized" : "Transcript"}</span>
        <span class="tr__hero-date">{row.date} · {row.dur}</span>
        <div style="margin-left:auto;display:flex;gap:8px">
          <sl-button size="small" variant="default" href={urls.exportMd(id)} download title="Export as Markdown">
            <sl-icon slot="prefix" name="download"></sl-icon> Export
          </sl-button>
          <sl-button
            size="small"
            variant="default"
            disabled={!session.canUndo}
            onclick={() => session.undo()}
            title="Undo last edit"
          >
            <sl-icon slot="prefix" name="arrow-counterclockwise"></sl-icon> Undo
          </sl-button>
          {#if !session.readonly}
            <button
              class="editmode"
              class:is-on={session.editMode}
              type="button"
              onclick={() => session.toggleEditMode()}
              title="Toggle edit mode"
            >
              {#if session.editMode}
                <span class="editmode__dot"></span> Editing
              {:else}
                <sl-icon name="pencil"></sl-icon> Edit transcript
              {/if}
            </button>
          {/if}
        </div>
      </div>

      <h1 class="tr__title">
        <span class="rec-name">{row.name}</span>
        <button
          class="rec-edit"
          type="button"
          title="Rename recording"
          aria-label="Rename recording"
          onclick={() => renameModal?.open(id, row.name)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/></svg>
        </button>
      </h1>

      <p class="tr__standfirst">
        {row.num_segments} segments · model <span class="mono">{row.model}</span>
        {#if row.language} · language <span class="mono">{row.language}</span>{/if}
        {#if !row.diarized} · no speaker labels{/if}.
      </p>
      <div class="tr__hr"></div>

      {#if session.editing}
        <div class="edit-hint">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/></svg>
          <span>
            <b>Reassign a passage.</b> Select any span of text inside a turn, then right-click
            it and choose <b>Reassign to speaker</b>. The turn splits in three — the words
            before and after stay with the original speaker, the selection moves to whoever
            you pick.
          </span>
        </div>
      {/if}

      {#if session.activeLang}
        <div class="tr__tx-note">
          <sl-icon name="translate"></sl-icon>
          <span>{session.nativeOf(session.activeLang)} · machine translation</span>
          <span class="muted">— segment-level sync only</span>
        </div>
        <div class="tr__downloads">
          <span class="dl-label">Download</span>
          {#each session.formats as fmt (fmt)}
            <a class="dl-chip" href={session.downloadUrl(fmt)} download>{fmt}</a>
          {/each}
        </div>
      {/if}

      <div class="tr__body">
        <TranscriptBody
          {currentTime}
          onseek={(t) => player?.seek(t)}
          onEditSpeaker={(key, label) => speakerModal?.open(key, label)}
        />
      </div>
    </div>

    <AudioPlayer bind:this={player} bind:currentTime src={urls.audio(id)} />
  </div>

  <AddTranslationModal bind:this={addModal} />
  <SpeakerRenameModal bind:this={speakerModal} />
  <RenameModal bind:this={renameModal} onsaved={(_id, name) => row && (row.name = name)} />
{/if}
