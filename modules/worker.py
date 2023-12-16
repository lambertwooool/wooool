import time
import threading
import traceback
import uuid
from modules import process_handler, process_upscale, process_reface, util

waiting_tasks = []
working_task = {}

def append(task):
	task["guid"] = str(uuid.uuid4())
	util.state(task, "new")
	util.output(task, 1, 'waiting ...')

	waiting_tasks.append(task)
	return task["guid"]

def stop(task_id):
     if task_id in working_task:
        task = working_task[task_id]
        process_handler.stop(task)

def skip(task_id):
     if task_id in working_task:
        task = working_task[task_id]
        process_handler.skip(task)

def worker():
    while True:
        time.sleep(0.01)

        if len(waiting_tasks) > 0:
            task = waiting_tasks.pop(0)
            working_task[task["guid"]] = task
            try:
                util.state(task, "start")
                util.output(task, 5, 'processing ...')
                action = task.get("action", "generate")
                if action == "generate":
                    process_handler.handler(task)
                elif action == "upscale":
                    process_upscale.handler(task)
                elif action == "reface":
                    process_reface.handler(task)
                else:
                    pass
            except Exception as e:
                traceback.print_exc()
                util.output(task, 100, "error", True)
                
            # util.state(task, "done")
    pass

threading.Thread(target=worker, daemon=True).start()

