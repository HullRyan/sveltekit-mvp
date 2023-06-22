import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/kit/vite";
import { mdsvex } from 'mdsvex';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			pages: "build",
			assets: "build",
			fallback: undefined,
			precompress: false,
			strict: true,
		}),
	},
	extensions: ['.svelte', '.md'],
	preprocess: [vitePreprocess(), mdsvex(
		{
			extensions: ['.md', '.html'],
		}
	)],
};
export default config;
