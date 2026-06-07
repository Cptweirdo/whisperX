<script lang="ts">
  import { session } from "../../lib/stores/session.svelte";

  let { onadd }: { onadd: () => void } = $props();

  let open = $state(false);
  const label = $derived(session.activeLang ? session.nativeOf(session.activeLang) : "Translate");
  const codes = $derived(Object.keys(session.translations));

  function choose(code: string) {
    open = false;
    if (!code) {
      session.showOriginal();
    } else if (session.translations[code]?.status === "done") {
      session.selectTranslation(code);
    }
  }

  function onWindowClick(e: MouseEvent) {
    if (open && !(e.target as HTMLElement).closest?.("#lang-pick")) open = false;
  }
</script>

<svelte:window onclick={onWindowClick} />

<div class="lang-pick" id="lang-pick">
  <button
    class="trans-chip"
    class:is-on={session.activeLang !== null}
    type="button"
    aria-haspopup="true"
    aria-expanded={open}
    onclick={(e) => {
      e.stopPropagation();
      open = !open;
    }}
  >
    <sl-icon name="translate"></sl-icon>
    <span>{label}</span>
    <sl-icon class="caret" name="chevron-down"></sl-icon>
  </button>

  {#if open}
    <div class="lang-menu">
      <div class="lang-menu__head">Translation</div>
      <button class="lang-opt" type="button" onclick={() => choose("")}>
        <div>
          <div class="lang-opt__native">{session.sourceLabel}</div>
          <div class="lang-opt__name">Original · spoken language</div>
        </div>
        {#if session.activeLang === null}
          <span class="lang-opt__check"><sl-icon name="check-lg"></sl-icon></span>
        {:else}
          <span class="lang-opt__tag">Source</span>
        {/if}
      </button>

      {#each codes as code (code)}
        {@const st = session.translations[code] || {}}
        <button class="lang-opt" type="button" onclick={() => choose(code)}>
          <div>
            <div class="lang-opt__native">{session.nativeOf(code)}</div>
            <div class="lang-opt__name">
              {st.status === "done"
                ? "Translated"
                : st.status === "running"
                  ? "In progress"
                  : st.error || "Translation failed"}
            </div>
          </div>
          {#if st.status === "done"}
            {#if session.activeLang === code}
              <span class="lang-opt__check"><sl-icon name="check-lg"></sl-icon></span>
            {:else}
              <span class="lang-opt__tag">{code.toUpperCase()}</span>
            {/if}
          {:else if st.status === "running"}
            <span class="lang-opt__stat running">
              <span class="spin"><sl-icon name="arrow-repeat"></sl-icon></span> Translating
            </span>
          {:else}
            <span class="lang-opt__stat error">
              <sl-icon name="exclamation-triangle"></sl-icon> Failed
            </span>
          {/if}
        </button>
      {/each}

      <button
        class="lang-opt lang-opt--enroll"
        type="button"
        onclick={() => {
          open = false;
          onadd();
        }}
      >
        <sl-icon name="plus-lg"></sl-icon> Add a translation…
      </button>
    </div>
  {/if}
</div>
