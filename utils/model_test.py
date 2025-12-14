from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
import json
import os
import time


def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    dic_agent_conf["EPSILON"] = 0
    dic_agent_conf["MIN_EPSILON"] = 0

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i)
        )
        agents.append(agent)
    try:
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf
        )
        total_time = dic_traffic_env_conf["RUN_COUNTS2"]
        state, step_time, list_need = env.reset()
        if "IS_CONTINOUS" in dic_traffic_env_conf.keys():
            is_continous = dic_traffic_env_conf["IS_CONTINOUS"]
        else:
            is_continous = False
        # testing_start_time = time.time()
        while step_time < total_time:
            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
                phase_action, duration_action = agents[i].choose_action(state, list_need, noise=False)
            if len(phase_action) == 1:
                duration_action = [duration_action]
            # 若该交叉口相位无变化，则绿灯时间可以从1秒开始
            # 若该交叉口相位有变化，则绿灯时间至少为黄灯
            if is_continous:
                idx = 0
                for i in list_need:
                    intersection = env.list_intersection[i]
                    if intersection.current_phase_index != (phase_action[idx] + 1):
                        duration_action[idx] = max(duration_action[idx], dic_traffic_env_conf["YELLOW_TIME"])
                    idx += 1
            next_state, step_time, list_need = env.step(phase_action, duration_action, is_continous)
            state = next_state
        env.batch_log_2()
        env.end_cityflow()
    except:
        print("============== error occurs in model_test ============")
