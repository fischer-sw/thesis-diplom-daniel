#!/usr/bin/python

import sys
import os
import subprocess
import logging
import json
import math

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")
"""
This script calculates the remaining runtime of ANSYS Fluent jobs
"""


def calc_runtime(jobs):
	for job in jobs:
		job_name = jobs[job]["name"]
		days = 0
		if "d" in jobs[job]["runtime"]:
			days = float(jobs[job]["runtime"].split("-")[0])
		run_time = days * 24 + float(jobs[job]["runtime"].split(":")[0]) + float(jobs[job]["runtime"].split(":")[1])/60.0
		# logging.info("Run time {}".format(run_time))
		estimated_runtime = jobs[job]["total_time"]/jobs[job]["flow_time"] * run_time
		remain = estimated_runtime - run_time
		# logging.info(remain)
		flow_time = jobs[job]["flow_time"]
		if remain < 24:
			logging.info("{} remaining time {}h current flow_time: {}s".format(job_name, round(remain, 2), flow_time))
		else:
			logging.info("{} remaining time {}h ->  {}d {}h current flow_time: {}s".format(job_name, round(remain, 2), math.floor(remain/24), round(remain%24, 2), flow_time))

		wall_time = float(jobs[job]["wall_time"].split(":")[0]) + float(jobs[job]["wall_time"].split(":")[1])/60.0
		wall_time_remain = wall_time - run_time
		if remain > wall_time_remain:
			logging.warning("{}, Remaining runtime {} > walltime {}".format(job_name, remain, wall_time_remain))
			logging.warning("Job {} : estimated total runtime: {} current walltime: {}".format(job_name, estimated_runtime, wall_time))


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
		
	# logging.info("Jobs : {}".format(jobs))
	calc_runtime(jobs)

if __name__ == "__main__":
	get_job_info()
