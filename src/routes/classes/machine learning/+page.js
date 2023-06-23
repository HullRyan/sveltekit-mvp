import { page } from '$app/stores';

export async function load() {
	let links = [];

	for (const [key, value] of Object.entries(import.meta.glob(`./*/*.md`))) {
		links.push(key.split('.')[1].split('/')[2]);
	}
	for (const [key, value] of Object.entries(import.meta.glob(`./*/*.html`, { as: 'raw', eager: true }))) {
		links.push(key.split('.')[1].split('/')[2]);
	}
	
	if (links?.length == undefined) {
		return {
			status: 404,
		};
	}

	return {
		links,
	};
}
