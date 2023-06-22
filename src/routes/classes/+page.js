import { page } from "$app/stores";

export async function load({ params }) {
	// Folders in the current directory, Set to not have duplicates
	let folders = [];

	// Get all folders in the current directory adding to the set
	for (const [key, value] of Object.entries(import.meta.glob(`./*/*.*`))) {
		if (folders.indexOf(key.split(".")[1].split("/")[1]) == -1) {
			folders.push(key.split(".")[1].split("/")[1]);
		}
	}

	// If none, return 404
	if (folders?.length == undefined) {
		return {
			status: 404,
		};
	}

	folders = [...folders];

	// Return the folder names
	return {
		folders,
	};
}
