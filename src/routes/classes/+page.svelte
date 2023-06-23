<script>
	import Links from "../../lib/Links.svelte";
	export let data;
</script>

<div>
	<Links links={data.folders} />
</div>
