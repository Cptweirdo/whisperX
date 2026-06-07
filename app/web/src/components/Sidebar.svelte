<script lang="ts">
  import { router, link } from "../lib/router.svelte";
  import { ui } from "../lib/stores/ui.svelte";
  import { models } from "../lib/stores/models.svelte";

  const active = $derived(router.current.name);
</script>

<aside class="sb">
  <a class="sb__brand" href="/" use:link>
    <div class="sb__logo"><img src="/favicon.svg" alt="Manuscript" /></div>
    <div>
      <div class="sb__name">Manuscript</div>
      <div class="sb__tag">WHISPERX · LOCAL</div>
    </div>
  </a>

  <button class="sb__new" onclick={() => ui.requestNewRecording()}>
    <sl-icon name="plus-lg"></sl-icon> New Recording
  </button>

  <nav class="sb__nav">
    <a
      class="sb__link"
      class:sb__link--active={active === "dashboard" || active === "transcript"}
      href="/"
      use:link
    >
      <sl-icon name="mic"></sl-icon> Recordings
    </a>
    <a class="sb__link" class:sb__link--active={active === "settings"} href="/settings" use:link>
      <sl-icon name="gear"></sl-icon> Settings
    </a>
  </nav>

  <div class="sb__spacer"></div>
  <div class="sb__user">
    <div class="avatar avatar--round">WX</div>
    <div>
      <div style="font-weight:600;font-size:15px">WhisperX</div>
      <div class="mono-meta">{models.deviceLabel} · LOCAL</div>
    </div>
  </div>
</aside>
