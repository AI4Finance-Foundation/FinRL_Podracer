import os
import shutil
import time

import numpy as np

"""run this process independently (single file)"""


def lock_before_io(path):
    lock_path = f"{path}.lock"

    while os.path.exists(lock_path):
        time.sleep(0.1)
    os.mknod(lock_path)


def lock_after_io(path):
    lock_path = f"{path}.lock"
    os.remove(lock_path)


def lock_copy(src_path, dst_path):
    lock_before_io(src_path)
    lock_before_io(dst_path)

    shutil.copytree(src_path, dst_path)
    lock_after_io(src_path)
    lock_after_io(dst_path)


def save_pod_info(info_path, pod_id, total_step, used_time, cumulative_rewards):
    # info_path = f"{pod_path}/pod_info.dict"
    with open(info_path, 'a') as f:
        info_dict = {
            'pod_id': pod_id,
            'total_step': total_step,
            'used_time': used_time,
            'cumulative_rewards_avg': np.average(cumulative_rewards),
            'cumulative_rewards_std': np.std(cumulative_rewards),
        }
        f.write(repr(info_dict))


def read_pod_info(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            info_dict = eval(f.readlines()[-1])
    else:
        info_dict = dict()
    return info_dict


def copy_top_n_to_pod(pod_paths, info_dicts, top_n=2):
    pod_rs = [info_dict['cumulative_rewards_avg'] for info_dict in info_dicts]
    pod_rs_sort = np.argsort(pod_rs)

    pod_i_top = pod_rs_sort[-top_n:]
    pod_i_bot = pod_rs_sort[:-top_n]

    for i, dst_i in enumerate(pod_i_bot):
        dst_path = pod_paths[dst_i]

        src_i = pod_i_top[i % top_n]
        src_path = pod_paths[src_i]

        lock_copy(src_path, dst_path)


def run():
    pod_num = 4
    # pod_num = 16
    # pod_num = 80

    dir_path = './PodRacerERL'

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    else:
        os.mkdir(dir_path)

    pod_names = [f'pod_{gpu_id:03}' for gpu_id in range(pod_num)]
    pod_paths = [f"{dir_path}/{pod_name}" for pod_name in pod_names]

    for pod_path in pod_paths:
        os.makedirs(pod_path)

    if_stop = False
    while not if_stop:
        if_stop = os.path.exists(f"{dir_path}/stop.sign")

        info_dicts = list()
        for pod_path in pod_paths:

            if_update = False
            while not if_update:
                if_update = os.path.exists(f"{pod_path}/update.sign")
                time.sleep(1)
            os.remove(f"{pod_path}/update.sign")

            info_dict = read_pod_info(f"{pod_path}/pod_info.dict")
            info_dicts.append(info_dict)

        copy_top_n_to_pod(pod_paths, info_dicts, top_n=2)


if __name__ == '__main__':
    run()
