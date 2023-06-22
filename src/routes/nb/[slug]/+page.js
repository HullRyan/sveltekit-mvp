export async function load({ params }) {
	let page, post;

	try {
		page = await import(`../${params.slug}.html?raw`);
	} catch (e) {
		post = await import(`../${params.slug}.md`);
	}

	let pageContent = page?.default;
	let postContent = post?.default;

	return {
		pageContent,
		postContent,
	};
}
