<script>
	import "../app.css";
    import Footer from "../lib/Footer.svelte";
    import NavigationBar from "../lib/NavigationBar.svelte";
</script>

<div class="flex flex-col h-screen">
    <NavigationBar />
    <div class="flex-grow">
        <slot />
    </div>
    <Footer />
</div>
