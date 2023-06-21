<script>
	export let description;
	let count = 0;
</script>

<div class="flex flex-col items-center justify-between">
	<div class="flex flex-col items-center gap-4 p-6 w-96 h-48 border rounded-lg">
		<div class="font-bold flex flex-col items-center justify-between p-1">
			Count: {count}
		</div>
		<button
			on:click={() => {
				count++;
			}}
			class="rounded-lg bg-slate-200 hover:bg-slate-300 p-2">Click Me</button
		>
		{#if count % 2 == 0}
			<div class="flex flex-col items-center justify-between p-1">
				{description}
			</div>
		{/if}
	</div>
</div>
