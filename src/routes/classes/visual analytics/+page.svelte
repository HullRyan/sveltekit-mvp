<script>
    import { page } from '$app/stores';
    export let data;
    let links = data.links;
</script>

<div>
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