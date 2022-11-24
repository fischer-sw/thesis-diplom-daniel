#!/usr/bin/python

import sys
import os
import subprocess
import logging
import json

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")

def store_jobs(jobs):
	tmp_jobs = list(jobs.keys())
	cfg_path = os.path.join("/net", "fileu", "user", "fische42", "Daniel", "thesis-diplom-daniel", "Software", "python", "conf.json")
	if os.path.exists(cfg_path):
		with open(cfg_path) as f:
			cfg = json.load(f)
		cfg["cases"] = tmp_jobs

		with open(cfg_path, "w") as f:
			json.dump(cfg, f, ensure_ascii=False, sort_keys=True ,indent=4)
		logging.info("Added {} cases to config".format(tmp_jobs))
	else:
		logging.warning("No config file at location {} exists".format(cfg_path))
		logging.warning("Files available are: {}".format(os.listdir(cfg_path)))

def get_job_info():
	res = subprocess.check_output(["qstat", "-u" ,"fische42"])
	raw_data = res[320:].split("\n")[1:]
	jobs = {}
	for line in raw_data:
		if len(line.split(" ")) < 3:
			continue
		char = line.split(" ")[1]
		tmp_line = str(line).strip(char).split(" ")
		job_cleaned = []
		for ele in tmp_line:
			if ele != char:
				job_cleaned.append(ele)

		if "R" in tmp_line:
			job = {
				"job_id" : job_cleaned[0],
				"user" : job_cleaned[1],
				"queue" : job_cleaned[2],
				"name": job_cleaned[3],
				"nodes": int(job_cleaned[5]),
				"tasks": int(job_cleaned[6]),
				"wall_time" : job_cleaned[8],
				"status" : job_cleaned[9],
				"runtime" : job_cleaned[10]
			}
			# logging.info("name : {}, tasks : {}, status: {}".format(job["name"],job["tasks"], job["status"]))
			if job["tasks"] != 1:
				cas_cfg_path = os.path.join(sys.path[0], job["name"], "cases.json")
				cas_path = os.path.join(sys.path[0], job["name"])
				if os.path.exists(cas_cfg_path) == False:
					logging.info("cases.json does not exsist for case" + job["name"] + "at" + cas_cfg_path)
				else:
					with open(cas_cfg_path) as f:
						cfg = json.load(f)
					# logging.info(cfg)
					job["total_time"] = cfg[job["name"]]["total_time"]
					jobs[job["name"]] = job
					files = os.listdir(cas_path)
					for file in files:
						if ".trn" in file:
							with open(os.path.join(cas_path, file)) as f:
								lines = f.readlines()[-20:]
							for line in lines:
								if "Flow time" in line:
									# logging.info(line)
									flow_time = float(line.split("= ")[1].split("s,")[0])
									# logging.info(flow_time)
									job["flow_time"] = flow_time
									break
		else:
			continue
		
	store_jobs(jobs)

if __name__ == "__main__":
	get_job_info()