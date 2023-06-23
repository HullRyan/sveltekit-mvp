<script>
    //This could be further improved by recursively checking the 
    //path to hand any level of subfolders
	import { page } from "$app/stores";
	$: path = $page.url.pathname;
</script>

<nav class="flex gap-10 p-4 border-b-2">
	<a href="/" class:font-bold={path == "/"}>Home</a>
	<a href="/about" class:font-bold={path == "/about"}>About</a>
	<span
		><a href="/classes" class:font-bold={path == "/classes"}>Classes</a>
		{#if decodeURI(path.split("/")[2]) != "undefined"}
			/ <a
				href="/{path.split('/')[1]}/{path.split('/')[2]}"
				class:font-bold={path.split("/")[3] == undefined}
			>
				{decodeURI(path.split("/")[2])}</a
			>
		{/if}
		{#if decodeURI(path.split("/")[3]) != "undefined"}
			/ <a href={path} class="font-bold"> {decodeURI(path.split("/")[3])}</a>
		{/if}
	</span>
</nav>
