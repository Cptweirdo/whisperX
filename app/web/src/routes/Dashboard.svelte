<script lang="ts">
  let { id }: { id?: string } = $props();
</script>

<div class="empty"><h2>Dashboard</h2>{#if id}<p>{id}</p>{/if}</div>
