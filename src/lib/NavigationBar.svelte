<script>
    import { page } from '$app/stores';
</script>

<nav class="flex gap-10 p-6 border-b-2">
    <a href="/" class:font-bold={$page.url.pathname == '/'}>Home</a>
    <a href="/about" class:font-bold={$page.url.pathname == '/about'}>About</a>
</nav>