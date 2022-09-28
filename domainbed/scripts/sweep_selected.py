"""
Run sweeps with selected hyperparameters
"""

import argparse
import copy
import hashlib
import json
import os
import shutil

from domainbed import datasets
from domainbed import algorithms
from domainbed import command_launchers

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args):
        self.train_args = copy.deepcopy(train_args)
        self.output_dir = self.train_args['output_dir']
        if self.train_args['algorithm'] in algorithms.ALGORITHMS:
            command = ['python', '-m', 'domainbed.scripts.train']
        else:
            command = ['python', '-m', 'domainbed.match.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            if isinstance(v, bool):
                if v:
                    command.append(f'--{k}')
            else:
                command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        # np.random.shuffle(jobs)
        # run jobs in order
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(dataset_names, clf_algorithms, matcher_algorithms, 
                    selected_records, model_selection_method, use_gpu, out_dir, 
                    data_dir, num_cf, test_envs, single_test_envs, hparams,
                    distance_func, use_oracle_match, balance_rate):
    args_list = []
    for dataset in dataset_names:
        if len(test_envs)>0:
            all_test_envs = test_envs
        elif single_test_envs:
            print("use single test env")
            all_test_envs = [
                [i] for i in range(datasets.num_environments(dataset))]
        else:
            all_test_envs = all_test_env_combinations(
                datasets.num_environments(dataset))
        
        for envs in all_test_envs:
            
            for match_algorithm in matcher_algorithms:
                matcher_ckpt_dir = os.path.join(out_dir, dataset, match_algorithm, 
                            'test_env' + '_'.join([str(env) for env in envs]))
                if not os.path.exists(matcher_ckpt_dir):
                    train_args = {}
                    train_args['dataset'] = dataset
                    train_args['algorithm'] = match_algorithm
                    train_args['test_envs'] = envs
                    train_args['holdout_fraction'] = 0
                    train_args['hparams_seed'] = 0
                    train_args['data_dir'] = data_dir
                    train_args['use_match'] = False
                    if use_gpu:
                        train_args['use_gpu'] = use_gpu
                    train_args['output_dir'] = matcher_ckpt_dir
                    args_list.append(train_args)

            for algorithm in clf_algorithms:
                for match_algorithm in matcher_algorithms:
                    matcher_ckpt_path = os.path.join(out_dir, dataset, match_algorithm, 
                                        'test_env' + '_'.join([str(env) for env in envs]), 'model.pkl')
                    for trial_seed in range(3):
                        record = selected_records[model_selection_method]\
                            [dataset][algorithm]["test_env_{}".format(envs[0])]\
                            [trial_seed]
                        train_args = record['args']
                        none_keys = []
                        for k, v in train_args.items():
                            if v is None:
                                none_keys.append(k)
                        for k in none_keys:
                            del train_args[k]
                        train_args['data_dir'] = data_dir
                        train_args['hparams'] = json.dumps(record['hparams'])
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        if use_gpu:
                            train_args['use_gpu'] = use_gpu
                        train_args['use_match'] = True
                        train_args['match_algorithm'] = match_algorithm
                        train_args['matcher_checkpoint'] = matcher_ckpt_path
                        train_args['distance_func'] = distance_func
                        args_str = json.dumps(train_args, sort_keys=True)
                        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
                        train_args['num_cf'] = num_cf
                        train_args['output_dir'] = os.path.join(out_dir, dataset, algorithm + '_' + match_algorithm + '_' + str(num_cf), args_hash)
                        args_list.append(train_args)

                if len(matcher_algorithms) == 0:
                    for trial_seed in range(3):
                        record = selected_records[model_selection_method]\
                            [dataset][algorithm]["test_env_{}".format(envs[0])]\
                            [trial_seed]
                        train_args = record['args']
                        none_keys = []
                        for k, v in train_args.items():
                            if v is None:
                                none_keys.append(k)
                        for k in none_keys:
                            del train_args[k]
                        train_args['data_dir'] = data_dir
                        train_args['hparams'] = json.dumps(record['hparams'])
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_str = json.dumps(train_args, sort_keys=True)
                        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
                        if use_gpu:
                            train_args['use_gpu'] = use_gpu
                        if use_oracle_match:
                            train_args['use_oracle_match'] = use_oracle_match
                            train_args['balance_rate'] = balance_rate
                        train_args['use_match'] = False
                        args_str = json.dumps(train_args, sort_keys=True)
                        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
                        train_args['output_dir'] = os.path.join(out_dir, dataset, algorithm, args_hash)
                        args_list.append(train_args)

    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=['ERM'])
    parser.add_argument('--match_algorithms', nargs='+', type=str, default=[])
    parser.add_argument('--selected_records', type=str, 
                        default='out/sweep_7df6f06/chosen_records_test.json')
    parser.add_argument('--model_selection_method', type=str, 
                        default='training-domain validation set')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--num_cf', type=int, default=1)
    parser.add_argument('--test_envs', nargs='+', type=str, default=[])
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--distance_func', type=str, default="kld")
    parser.add_argument('--use_oracle_match', action='store_true')
    parser.add_argument('--balance_rate', type=float, default=1.0)
    args = parser.parse_args()

    args_list = make_args_list(
        dataset_names=args.datasets,
        clf_algorithms=args.algorithms,
        matcher_algorithms=args.match_algorithms,
        selected_records=json.loads(open(args.selected_records).read()),
        model_selection_method=args.model_selection_method,
        use_gpu=args.use_gpu,
        data_dir=args.data_dir,
        out_dir=args.output_dir,
        num_cf = args.num_cf,
        test_envs=args.test_envs,
        single_test_envs=args.single_test_envs,
        distance_func=args.distance_func,
        use_oracle_match=args.use_oracle_match,
        balance_rate=args.balance_rate,
        hparams=args.hparams,
    )

    jobs = [Job(train_args) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)