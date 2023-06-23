<script>
	//This could be further improved by recursively checking the
	//path to hand any level of subfolders. Would be better to 
    //use env. variables to set the base path and baseNum
	import { page } from "$app/stores";
	$: path = $page.url.pathname;
	$: basePath = (path).includes("/sveltekit-mvp")
		? "/sveltekit-mvp"
		: "";
    $: baseNum = (path).includes("/sveltekit-mvp") ? 2 : 1;
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
		{#if decodeURI(path.split("/")[baseNum+ 1 ]) != "undefined"}
			/ <a
				href="{basePath + '/' + path.split('/')[baseNum]}/{path.split('/')[baseNum + 1]}"
				class:font-bold={path.split("/")[baseNum + 2] == undefined}
			>
				{decodeURI(path.split("/")[baseNum + 1])}</a
			>
		{/if}
		{#if decodeURI(path.split("/")[baseNum + 2]) != "undefined"}
			/ <a href={path} class="font-bold"> {decodeURI(path.split("/")[baseNum + 2])}</a>
		{/if}
	</span>
</nav>
