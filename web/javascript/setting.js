function save_ui_settings(data) {
	data = JSON.parse(data);
	for (let x in data) {
		if (["mc", "mc_other", "prompt_main"].indexOf(x) > -1 || /^refer_lora_/.test(x)) {
			delete data[x]
		}
	}
	let jsonStr = JSON.stringify(data);
	localStorage["setting"] = jsonStr;
}

function get_ui_settings(data) {
	document.getElementById("btn_base_model_info").target = "_blank";
	return localStorage["setting"] || data;
}

function model_link(model) {
	document.getElementById("btn_base_model_info").target = "_blank";
	return model;
}

function download_ui_settings(data) {
	data = JSON.parse(data);
	["mc", "mc_other", "prompt_main", "ckb_pro"].forEach(x => delete data[x])
	let jsonStr = JSON.stringify(data, null, 4);
	let blob = new Blob([jsonStr], {
  		type: "application/json"
	});

	let a = document.createElement("a");
	a.href = URL.createObjectURL(blob);
	a.setAttribute("download", "wooool.json");
	a.click();
}

function reset_ui_settings(data) {
	delete localStorage["setting"];
}