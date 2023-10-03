import importlib
import multiprocessing
import random
from enum import Enum

import gym

from experimental.urdf_name_resolver import get_urdf_filepath, get_urdf_folderpath

NUM_WORKERS = 5
class HandoverObject(Enum):
    PILL = "pill"
    CUP = "cup"
    CANE = "cane"


    @staticmethod
    def from_string(label):
        if label == "pill":
            return HandoverObject.PILL
        elif label == "cup":
            return HandoverObject.CUP
        elif label == "cane":
            return HandoverObject.CANE
        else:
            raise ValueError(f"Invalid handover object label: {label}")


objectTaskMapping = {
        HandoverObject.PILL: "comfort_taking_medicine",
        HandoverObject.CUP: "comfort_drinking",
        HandoverObject.CANE: "comfort_standing_up"
}

def make_env(config):
    env_name, person_id, smpl_file, object_name, coop, seed = config
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    env.set_smpl_file(smpl_file)

    human_urdf_path = get_urdf_filepath(get_urdf_folderpath(person_id))
    env.set_human_urdf(human_urdf_path)

    task = get_task_from_handover_object(object_name)
    env.set_task(task)
    env.render()
    env.reset()
    return env
def get_task_from_handover_object(object_name):
    if not object_name:
        return None
    object_type = HandoverObject.from_string(object_name)
    task = objectTaskMapping[object_type]
    return task


def make_headless_envs(env_name, person_id, smpl_file, handover_obj, coop = True): # headless
    envs = [None] * NUM_WORKERS
    for i in range (NUM_WORKERS):
        envs[i]= make_env(env_name, person_id, smpl_file, handover_obj, coop)
        # envs.append(env)
        # print ('env_id: ', env.id)
    for i in range (NUM_WORKERS):
        envs[i].reset()
    return envs

def create_env_processes(NUM_WORKERS):
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    configs = []
    for i in range(NUM_WORKERS):
        configs.append(('HumanComfort-v1', 'p001', 'examples/data/slp3d/p001/s01.pkl', 'pill', True, 1001))
    results = pool.map(make_env, configs)
    for num, result in enumerate(results):
        print('Done training for {} {}'.format(num, result.id))

import multiprocessing

class WorkerProcess(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, env_config):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.env_config = env_config
        self.env = None
        # self.env = make_env(env_config)

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:  # Sentinel value to indicate termination
                break
            result = self.perform_task(task)
            self.result_queue.put(result)

    def perform_task(self, task):
        if not self.env:
            self.env = make_env(self.env_config)
        return task*10

# def main():
#     num_processes = 2
#     tasks = [1, 2, 3, 4]
#
#     task_queue = multiprocessing.Queue()
#     result_queue = multiprocessing.Queue()
#
#     # Start worker processes
#     workers = [WorkerProcess(task_queue, result_queue) for _ in range(num_processes)]
#     for w in workers:
#         w.start()
#
#     # Enqueue tasks
#     for task in tasks:
#         task_queue.put(task)
#
#     # Collect results
#     for _ in tasks:
#         print(result_queue.get())
#
#     # Signal workers to terminate
#     for _ in range(num_processes):
#         task_queue.put(None)
#
#     # Wait for workers to complete
#     for w in workers:
#         w.join()
#
# if __name__ == "__main__":
#     main()


if __name__ == '__main__':
    # import concurrent.futures
    # envs = []
    configs = []
    for i in range (NUM_WORKERS):
        configs.append(('HumanComfort-v1', 'p001', 'examples/data/slp3d/p001/s01.pkl', 'pill', True, 1001))
    # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     results = executor.map(make_env, configs)
    # for num, result in enumerate(results):
    #     print('Done training for {} {}'.format(num, result.id))

    # envs = make_headless_envs('HumanComfort-v1', 'p001', 'examples/data/slp3d/p001/s01.pkl', 'pill', coop=False)

    # create_env_processes(NUM_WORKERS)

        num_processes = 2
        tasks = [1, 2, 3, 4]

        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        # Start worker processes
        workers = [WorkerProcess(task_queue, result_queue, configs[0]) for _ in range(num_processes)]
        for w in workers:
            w.start()

        # Enqueue tasks
        for task in tasks:
            task_queue.put(task)

        # Collect results
        for _ in tasks:
            print(result_queue.get())

        tasks = [6,7,8,9,10]
        # Enqueue tasks
        for task in tasks:
            task_queue.put(task)
        for _ in tasks:
            print(result_queue.get())
        # Signal workers to terminate
        # for _ in range(num_processes):
        #     task_queue.put(None)

        # Wait for workers to complete
        for w in workers:
            w.join()