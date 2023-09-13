# VERY hacky script but hey, gets the job done
import libtmux

task_names = [
    # "bernoulli_glm",
    # "gaussian_linear_uniform",
    # "gaussian_linear",
    # "gaussian_mixture",
    "lotka_volterra",
    # "sir",
    # "slcp_distractors",
    "slcp",
    # "two_moons",
]

server = libtmux.Server()

cuda_gpus = [1,2,3,4,5,6,7]
for task_idx, task_name in enumerate(task_names):
    server.new_session(attach=False)
    session = server.sessions[-1]
    p = session.attached_pane
    p.send_keys("conda activate chig", enter=True)
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_gpus[task_idx % len(cuda_gpus)]} python csi.py --task {task_name}"
    p.send_keys(cmd, enter=True)
    print(f"Launched: {cmd}")