<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../lib/api";
  import { router } from "../lib/router.svelte";
  import { settings } from "../lib/stores/settings.svelte";
  import { models } from "../lib/stores/models.svelte";
  import { notify } from "../lib/stores/toast.svelte";
  import BackupCard from "../components/settings/BackupCard.svelte";
  import type { OnboardingData } from "../lib/types";

  let data = $state<OnboardingData | null>(null);
  let step = $state(0);
  let token = $state("");
  let model = $state("");
  let device = $state("cpu");
  let verify = $state<{ ok: boolean | null; text: string } | null>(null);
  let verifying = $state(false);
  let finishing = $state(false);

  const STEP_NAMES = [
    ["01", "Welcome", "What Manuscript does"],
    ["02", "Access", "Hugging Face token"],
    ["03", "Backups", "Cloud backup · optional"],
    ["04", "Engine", "Model & backend"],
  ];

  const BACKENDS = [
    ["cpu", "CPU", "Runs on any machine — no GPU required. Slowest for long files.", "Universal · int8"],
    ["cuda", "CUDA", "NVIDIA GPUs. Highest throughput using float16 compute.", "NVIDIA · float16"],
    ["mlx", "MLX", "Apple Silicon (M-series). Native on-device acceleration for macOS.", "Apple Silicon"],
    ["whispercpp", "whisper.cpp", "whisper.cpp via pywhispercpp. Metal on Apple Silicon, CPU elsewhere. Fastest on Mac for large models.", "Metal · CPU"],
  ];
  function available(id: string): boolean {
    const m = data?.models;
    if (id === "cpu") return true;
    if (id === "cuda") return !!m?.cuda_available;
    if (id === "mlx") return !!m?.mlx_available;
    if (id === "whispercpp") return !!m?.whispercpp_available;
    return false;
  }

  const progress = $derived(step >= 4 ? 100 : ((Math.min(step, 3) + 1) / 4) * 100);
  const sizeNote = $derived(data?.sizes?.find((s: any) => s.id === model)?.note ?? "");
  const sizeName = $derived(data?.sizes?.find((s: any) => s.id === model)?.name ?? model);
  const backendName = $derived(BACKENDS.find((b) => b[0] === device)?.[1] ?? device);

  onMount(async () => {
    data = await api.onboarding.get();
    token = data.token ?? "";
    model = data.selected_size ?? "";
    device = data.models?.device ?? "cpu";
  });

  async function continueAccess() {
    if (!token.trim()) {
      step = 2;
      return;
    }
    verifying = true;
    verify = { ok: null, text: "Verifying…" };
    try {
      const r = await api.onboarding.verify(token.trim());
      verify = { ok: r.ok, text: r.detail };
      if (r.ok) step = 2;
    } catch {
      verify = { ok: false, text: "Could not reach the server. Try again." };
    } finally {
      verifying = false;
    }
  }

  async function finish() {
    finishing = true;
    try {
      const r = await api.onboarding.finish({ token: token.trim(), model, device });
      if (r.ok) {
        await Promise.all([settings.load(), models.load()]);
        router.navigate("/", { replace: true });
      } else {
        notify(r.error || r.store_error || "Could not finish setup.", "danger");
        if (r.store_error || r.error) step = 1;
      }
    } catch (e: any) {
      notify(e?.message || "Could not finish setup.", "danger");
    } finally {
      finishing = false;
    }
  }
</script>

<div class="ob">
  <div class="ob__cover">
    <div class="ob__brand">
      <div class="ob__logo"><img src={import.meta.env.BASE_URL + "favicon.svg"} alt="Manuscript" /></div>
      <div>
        <div class="ob__wordmark">Manuscript</div>
        <div class="ob__coverTag">FIRST-RUN SETUP</div>
      </div>
    </div>
    <div class="ob__prog"><i style={`width:${progress}%`}></i></div>

    <div class="ob__steps">
      {#each STEP_NAMES as [num, t, desc], i (num)}
        <div class="ob__step" class:ob__step--active={i === Math.min(step, 3) && step < 4} class:ob__step--done={i < step}>
          <div class="ob__stepNum">{num}</div>
          <div class="ob__stepBody"><div class="t">{t}</div><div class="d">{desc}</div></div>
          <div class="ob__stepDone"><sl-icon name="check-lg"></sl-icon></div>
        </div>
      {/each}
    </div>
    <div class="ob__colophon">WhisperX engine · local-first<br />Your audio never leaves this device.</div>
  </div>

  <div class="ob__pane">
    {#if step === 0}
      <section class="ob__step-panel is-active">
        <div class="ob__kicker">STEP 01 / 04 · WELCOME</div>
        <h1 class="ob__title">A quieter way to<br />turn talk into text.</h1>
        <p class="ob__lede">Manuscript transcribes, separates speakers, and lets you annotate long-form audio — built for researchers, journalists, and anyone who reads closely. Three quick steps and you're set.</p>
        <div class="ob__content">
          <div class="ob__feats">
            <div class="ob__feat"><div class="ob__featNum">01</div><div class="ob__featName">Transcribe</div><div class="ob__featDesc">Word-level timestamps from hours of audio, aligned for precise scrubbing.</div></div>
            <div class="ob__feat"><div class="ob__featNum">02</div><div class="ob__featName">Diarize</div><div class="ob__featDesc">Automatic speaker separation attributes every line to who said it.</div></div>
            <div class="ob__feat"><div class="ob__featNum">03</div><div class="ob__featName">Annotate</div><div class="ob__featDesc">Read, highlight, and edit the transcript in sync with playback.</div></div>
          </div>
        </div>
        <div class="ob__foot">
          <div class="spacer"></div>
          <button type="button" class="btn-primary lg" onclick={() => (step = 1)}>Begin setup <sl-icon name="arrow-right"></sl-icon></button>
        </div>
      </section>
    {:else if step === 1}
      <section class="ob__step-panel is-active">
        <div class="ob__kicker">STEP 02 / 04 · ACCESS</div>
        <h1 class="ob__title">Connect Hugging Face.<br /><span style="font-weight:400;opacity:.6">Optional.</span></h1>
        <p class="ob__lede">Speaker diarization works out of the box — the pyannote model is bundled with Manuscript. Adding a free Hugging Face token is <strong>optional</strong>; it only lets you fetch model updates later.</p>
        <div class="ob__content">
          <span class="ob__label">Access Token <span style="font-weight:400;opacity:.55">— optional</span></span>
          <div class="ob__tokenField">
            <sl-input type="password" password-toggle placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
              value={token} autocomplete="off" onsl-input={(e: any) => { token = e.target.value; verify = null; }}>
              <sl-icon slot="prefix" name="key"></sl-icon>
              <span slot="suffix" class="ob__inputTag">read access</span>
            </sl-input>
          </div>
          <div class="ob__hint">Stored in your operating system's secure keyring — never written to disk in plaintext, never uploaded.</div>
          {#if verify}
            <div class="ob__verify">
              <span class="frag {verify.ok === null ? '' : verify.ok ? 'frag--ok' : 'frag--err'}">
                <sl-icon name={verify.ok === null ? "hourglass-split" : verify.ok ? "check-circle" : "exclamation-triangle"}></sl-icon>
                {verify.text}
              </span>
            </div>
          {/if}
          <div class="ob__help">
            <div class="ob__helpH"><sl-icon name="question-circle"></sl-icon> How to get your token</div>
            <ol class="ob__ol">
              <li>Open <a class="ob__link" href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens</a> and sign in.</li>
              <li>Click <span class="ob__code">New token</span>, give it a name, and choose the <span class="ob__code">Read</span> role.</li>
              {#if data?.diarize_model}<li>Accept the user conditions on <a class="ob__link" href={`https://huggingface.co/${data.diarize_model}`} target="_blank" rel="noopener"><span class="ob__code">{data.diarize_model}</span></a>.</li>{/if}
              <li>Copy the token (starts with <span class="ob__code">hf_</span>) and paste it above.</li>
            </ol>
          </div>
        </div>
        <div class="ob__foot">
          <button type="button" class="btn-text lg" onclick={() => (step = 0)}>Back</button>
          <div class="spacer"></div>
          <button type="button" class="btn-primary lg" disabled={verifying} onclick={continueAccess}>Continue <sl-icon name="arrow-right"></sl-icon></button>
        </div>
      </section>
    {:else if step === 2}
      <section class="ob__step-panel is-active">
        <div class="ob__kicker">STEP 03 / 04 · BACKUPS</div>
        <h1 class="ob__title">Back up to the cloud.<br /><span style="font-weight:400;opacity:.6">Optional.</span></h1>
        <p class="ob__lede">Sync finished transcripts to a cloud account so they're safe and available across devices. Everything stays local first — backups are an extra copy you control.</p>
        <div class="ob__content">
          <div class="ob__backupBox"><BackupCard onboarding /></div>
        </div>
        <div class="ob__foot">
          <button type="button" class="btn-text lg" onclick={() => (step = 1)}>Back</button>
          <div class="spacer"></div>
          <button type="button" class="btn-text lg" onclick={() => (step = 3)}>Skip for now</button>
          <button type="button" class="btn-primary lg" onclick={() => (step = 3)}>Continue <sl-icon name="arrow-right"></sl-icon></button>
        </div>
      </section>
    {:else if step === 3}
      <section class="ob__step-panel is-active">
        <div class="ob__kicker">STEP 04 / 04 · ENGINE</div>
        <h1 class="ob__title">Pick your engine.</h1>
        <p class="ob__lede">Choose a model size and the backend that matches your hardware. You can change both later in Settings.</p>
        <div class="ob__content">
          <div class="ob__group">
            <div class="ob__groupH"><span class="h">Model Size</span><span class="meta">accuracy ↑ · speed ↓</span></div>
            <div class="ob__sizes">
              {#each data?.sizes ?? [] as s (s.id)}
                <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
                <div class="ob__size" class:ob__size--on={model === s.id} onclick={() => (model = s.id)}>
                  <div class="ob__sizeName">{s.name}</div>
                  <div class="ob__sizeMeta">{s.meta}</div>
                </div>
              {/each}
            </div>
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            <div class="ob__sizeNote">{@html sizeNote}</div>
          </div>

          <div class="ob__group">
            <div class="ob__groupH"><span class="h">Backend</span><span class="meta">compute device</span></div>
            <div class="ob__backends">
              {#each BACKENDS as [id, name, desc, tag] (id)}
                {@const avail = available(id)}
                <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
                <div class="ob__backend" class:ob__backend--on={device === id} class:ob__backend--off={!avail}
                  onclick={() => avail && (device = id)}>
                  <div class="ob__beTop"><span class="ob__beName">{name}</span><span class="ob__radio"></span></div>
                  <div class="ob__beDesc">{desc}{#if !avail} <em>Not available on this host.</em>{/if}</div>
                  <div class="ob__beTag">{tag}</div>
                </div>
              {/each}
            </div>
          </div>
        </div>
        <div class="ob__foot">
          <button type="button" class="btn-text lg" onclick={() => (step = 2)}>Back</button>
          <div class="spacer"></div>
          <button type="button" class="btn-primary lg" onclick={() => (step = 4)}>Finish setup <sl-icon name="check-lg"></sl-icon></button>
        </div>
      </section>
    {:else}
      <section class="ob__step-panel is-active">
        <div class="ob__done">
          <div class="ob__check"><sl-icon name="check-lg"></sl-icon></div>
          <h1 class="ob__doneTitle">You're all set.</h1>
          <p class="ob__doneLede">Manuscript is configured and ready. Drop in your first recording to generate a transcript.</p>
          <div class="ob__summary">
            <div class="ob__summaryItem"><div class="k">Token</div><div class="v">{token.trim() ? "hf_••••" + token.trim().slice(-4) : "Bundled model — no token"}</div></div>
            <div class="ob__summaryItem"><div class="k">Model</div><div class="v">{sizeName}</div></div>
            <div class="ob__summaryItem"><div class="k">Backend</div><div class="v">{backendName}</div></div>
          </div>
          <div style="margin-top:40px">
            <button type="button" class="btn-primary lg" disabled={finishing} onclick={finish}>Enter Manuscript <sl-icon name="arrow-right"></sl-icon></button>
          </div>
        </div>
      </section>
    {/if}
  </div>
</div>
