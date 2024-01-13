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

function endless(is_endless) {
	if (is_endless) {
		document.getElementById("btn_generate").click();
	}
	return is_endless;
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

function deleteSample(gl_sample_list, num_selected_sample, num_page_sample) {
	pic_item = gl_sample_list[num_selected_sample]
	if (pic_item && (event.shiftKey || window.confirm("Do you want to delete picture?\n" + pic_item[1] + "\n\n(Shift + Delete skip confirm)"))) {
		return [gl_sample_list, num_selected_sample, num_page_sample];
	} else {
		return [null, -1, 0];
	}
}