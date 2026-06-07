<script lang="ts">
  let { summary }: { summary: any } = $props();
</script>

<aside class="summary">
  <div class="summary__h">Library Summary</div>
  <div class="summary__row"><span>Recordings</span><span>{summary.count}</span></div>
  <div class="summary__row"><span>Transcribed</span><span>{summary.transcribed}</span></div>
  <div class="summary__row"><span>Total Audio</span><span>{summary.total_audio}</span></div>
  <div class="summary__bar"><i style={`width:${summary.pct}%`}></i></div>
  <div class="summary__quota">{summary.pct}% transcribed</div>
</aside>
