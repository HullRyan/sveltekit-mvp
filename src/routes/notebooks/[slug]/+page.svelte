<script>
    export let data;
</script>

<div class="p-16">
    {#if data.pageContent}
        {@html data.pageContent}
    {:else}
        <svelte:component this={data.postContent} />
    {/if}
</div>