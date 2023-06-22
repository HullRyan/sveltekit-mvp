import { marked } from 'marked';

export async function load({ params }) {
	const post = await import(`../${params.slug}.md`);
	const content = marked(post.default);

	return {
		content,
	};
}
