import os
import sys
import shutil
import yaml
import json
import argparse
from copy import deepcopy
from filelock import FileLock
import numpy as np
from hyperopt import fmin, tpe
from hyperopt.mongoexp import MongoTrials


PATH_OPT_CONFIGS = "../configs/optimization"


class ExperimentRunner:

    def __init__(
        self,
        default_config,
        results_dir,
        tmp_dir,
        python_path,
        train_path,
        debug=False,
        num_gpus=2
    ):
        self.default_config = default_config
        self.num_gpus = num_gpus
        self.results_dir = results_dir
        self.tmp_dir = tmp_dir
        self.python_path = python_path
        self.train_path = train_path
        self.debug = debug
        self.lockfile = os.path.join(tmp_dir, "gpu_lock")
        #print("\n\n\n", os.getcwd(), "\n\n\n")

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        open(self.lockfile, 'a').close()

    def make_trial_config(self, params):
        config = deepcopy(self.default_config)
        for key, value in params.items():
            cur_dct = config
            param_keys = key.split(".")
            for num_cur_key, cur_key in enumerate(param_keys):
                if num_cur_key+1 < len(param_keys):
                    if cur_key not in cur_dct:
                        cur_dct[cur_key] = {}
                    cur_dct = cur_dct[cur_key]
                else:
                    cur_dct[cur_key] = value
        return config

    def reserve_gpu(self):
        with FileLock(self.lockfile + ".lock"):
            with open(self.lockfile, "r+") as f:
                used_gpus =  f.readlines()
                used_gpus = list(map(int, used_gpus))
                free_gpus = [i for i in range(self.num_gpus) if i not in used_gpus]
                reserve_gpu = min(free_gpus)
                add_str = "\n" if len(used_gpus)>0 else ""
                f.write(add_str + str(reserve_gpu))
        return reserve_gpu

    def free_gpu(self, reserved_gpu):
        with FileLock(self.lockfile + ".lock"):
            with open(self.lockfile, "r+") as f:
                used_gpus =  f.readlines()
                used_gpus = list(map(int, used_gpus))
                used_gpus.remove(reserved_gpu)
                f.seek(0)
                f.write("\n".join(map(str, used_gpus)))
                f.truncate()

    def __call__(self, params):
        #print("\n\n\n", os.getcwd(), "\n\n\n")
        print("FGJEHFEYFBWIFKYBWEFGVBWEFKWEFVYWETFVEWF")
        config = self.make_trial_config(params)
        config_name_base = str(hash(str(config)))[:10]  # assumes persist order
        #if self.debug:
        #    config_name_base += "_debug"
        config_name = config_name_base + ".yml"
        config_path = os.path.join(self.tmp_dir, config_name)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)#, default_flow_style=False)

        gpu_num = self.reserve_gpu()
        run_str = f"CUDA_VISIBLE_DEVICES={gpu_num} {self.python_path} {self.train_path}"
        run_str += f" --config {config_path} --cv --logdir {self.tmp_dir}" 
        if self.debug:
            run_str += " --debug"
        os.system(run_str)

        config_name_base += "_debug" if self.debug else ""
        result_path = os.path.join(self.tmp_dir, config_name_base,
                                   "avg_metrics.json")
        with open(result_path, "r") as f:
            result = json.loads(f.read())
        best_epoch_score = np.max([v for k, v in result.items() if k!="best"])
        best_score = result["best"]

        self.free_gpu(gpu_num)
        shutil.rmtree(os.path.join(self.tmp_dir, config_name_base))
        new_config_name = str(int(best_score*10000)) + "_" + \
                          str(int(best_epoch_score*10000)) + "_" + \
                          config_name_base + ".yml"
        new_config_path = os.path.join(self.results_dir, new_config_name)
        os.rename(config_path, new_config_path)

        return -best_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-config', type=str)
    parser.add_argument('--space-config', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--results-dir', type=str, default="results")
    parser.add_argument('--tmp-dir', type=str, default="tmp")
    parser.add_argument('--exp-name', type=str, default="test")
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--num-trials', type=int, default=400)

    args = parser.parse_args()
    args.results_dir = os.path.join(args.results_dir, args.exp_name)
    return args


def replace_path_config(dct, key_replace=('df_path', 'pretrained_model_name_or_path')):
    for k, v in dct.items():
        if isinstance(v, dict):
            replace_path_config(dct[k])
        elif k in key_replace:
            dct[k] = os.path.abspath(dct[k])
    return dct


def main():
    sys.path.insert(0, PATH_OPT_CONFIGS)
    args = parse_args()
    with open(args.default_config, 'r') as stream:
        default_config = yaml.load(stream, Loader=yaml.FullLoader)
    default_config = replace_path_config(default_config)
    space = __import__(args.space_config[:-3]).space

    exp = ExperimentRunner(
        default_config=default_config,
        results_dir=os.path.abspath(args.results_dir),
        tmp_dir=os.path.abspath(args.tmp_dir),
        python_path=sys.executable,
        train_path=os.path.abspath(os.path.join(os.getcwd(), "..", "src", "train.py")),
        debug=args.debug,
        num_gpus=args.num_gpus
    )

    trials = MongoTrials('mongo://localhost:1234/tweet_sent/jobs', exp_key=args.exp_name)
    best = fmin(exp, space, trials=trials, algo=tpe.suggest, max_evals=args.num_trials)

if __name__=="__main__":
    main()
