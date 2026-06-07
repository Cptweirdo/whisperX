<script lang="ts">
  let { id }: { id?: string } = $props();
</script>

<div class="empty"><h2>Sidebar</h2>{#if id}<p>{id}</p>{/if}</div>
