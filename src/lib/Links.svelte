<script>
	import { page } from "$app/stores";
	export let links;
</script>

<div class="flex flex-col p-12 content-center justify-center text-center">
	<h1 class="text-2xl font-bold pb-4 capitalize">
		{decodeURI($page.url.pathname.split("/").slice(-1))}
	</h1>
	<ul>
		{#each links as link, i}
			<li>
				<a href="{$page.url.pathname}/{link}">
					{link}
				</a>
			</li>
		{/each}
	</ul>
</div>
