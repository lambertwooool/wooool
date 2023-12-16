function save_ui_settings(data) {
	localStorage["setting"] = data;
}

function get_ui_settings(data) {
	return localStorage["setting"] || data;
}