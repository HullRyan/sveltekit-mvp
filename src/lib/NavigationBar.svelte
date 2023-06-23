<script>
	//This could be further improved by recursively checking the
	//path to hand any level of subfolders
	import { page } from "$app/stores";
	let basePath = $page.url.pathname.contains("/sveltekit-mvp")
		? "/sveltekit-mvp"
		: "/";
	$: path = $page.url.pathname;
</script>

<nav class="flex gap-10 p-4 border-b-2">
	<a href={basePath} class:font-bold={path == basePath}>Home</a>
	<a href={basePath + "/about"} class:font-bold={path == basePath + "/about"}
		>About</a
	>
	<span
		><a
			href={basePath + "/classes"}
			class:font-bold={path == basePath + "/classes"}>Classes</a
		>
		{#if decodeURI(path.split("/")[3]) != "undefined"}
			/ <a
				href="{basePath + '/' + path.split('/')[2]}/{path.split('/')[3]}"
				class:font-bold={path.split("/")[4] == undefined}
			>
				{decodeURI(path.split("/")[3])}</a
			>
		{/if}
		{#if decodeURI(path.split("/")[4]) != "undefined"}
			/ <a href={path} class="font-bold"> {decodeURI(path.split("/")[4])}</a>
		{/if}
	</span>
</nav>
