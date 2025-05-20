import torch

import time

# import gymnasium
import fire
import tqdm

from controller.mppi import MPPI
from envs.navigation_2d import Navigation2DEnv

torch.set_num_threads(1)

def main(save_mode: bool = False):
    device = torch.device("cpu")
    env = Navigation2DEnv(device=device, min_distance=0.3)

    # solver
    solver = MPPI(
        horizon=8,
        num_samples=3000,
        dim_state=3,
        dim_control=3,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.tensor([0.5, 0.5, 0.5]),
        lambda_=1.0,
        auto_lambda=True,
        device=device
    )

    state = env.reset()
    max_steps = 500
    total_time = 0.0
    step_count = 0
    for i in range(max_steps):
        start = time.time()
        action_seq, state_seq = solver.forward(state=state)
        end = time.time()
        
        print(f"FPS: {1/(end-start):.2f}")
        
        total_time += end - start
        step_count += 1

        state, is_goal_reached = env.step(action_seq[0, :])

        is_collisions = env.collision_check(state=state_seq)

        top_samples, top_weights = solver.get_top_samples(num_samples=300)

        if save_mode:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="rgb_array",
            )
            # progress bar
            if i == 0:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        else:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="human",
            )
        if is_goal_reached:
            print("Goal Reached!")
            break

    average_time = total_time / step_count
    print("average solve time: {:.3f} ms".format(average_time * 1000))
    env.close()  # close window and save video if save_mode is True


if __name__ == "__main__":
    fire.Fire(main)
