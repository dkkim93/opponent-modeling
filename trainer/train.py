import copy
import numpy as np
from trainer.utils import *

total_step_count = 0
total_ep_count = 0
session_step_count = 0
session_ep_count = 0
TRAIN_REWARD_N = []
EVAL_REWARD_N = []


def eval_progress(env, student_n, log, tb_writer, args, log_result):
    eval_reward = 0.

    for i_eval in range(args.n_eval):
        env_obs_n = env.reset()
        ep_timesteps = 0.
        terminal = False

        while True:
            student_obs_n = get_student_obs(env_obs_n, ep_timesteps, args)
            student_action_n = []
            for student, student_obs in zip(student_n, student_obs_n):
                student_action = student.select_deterministic_action(np.array(student_obs))
                student_action_n.append(student_action)

            new_env_obs_n, reward_n, done_n, _ = env.step(copy.deepcopy(student_action_n))
            if ep_timesteps + 1 == args.ep_max_timesteps:
                terminal = True

            # For next timestep
            env_obs_n = new_env_obs_n
            eval_reward += reward_n[0]
            ep_timesteps += 1

            if terminal:
                break

    eval_reward /= args.n_eval

    if log_result:
        log[args.log_name].info("Evaluation Reward {:.5f} at episode {}".format(eval_reward, total_ep_count))
        tb_writer.add_scalar("reward/eval_reward", eval_reward, total_ep_count)

    return eval_reward


def collect_one_traj(student_n, teacher_n, asker_n, env, log, args, tb_writer):
    global total_step_count, total_ep_count, session_step_count, session_ep_count

    ep_reward = 0
    ep_timesteps = 0
    terminal = False
    env_obs_n = env.reset()

    while True:
        # Student selects its action
        student_obs_n = get_student_obs(env_obs_n, ep_timesteps, args)
        student_action_n = []
        for student, student_obs in zip(student_n, student_obs_n):
            student_action = student.select_stochastic_action(np.array(student_obs), session_step_count)
            student_action_n.append(student_action)

        # Get meta action
        # teacher_obs_n = [teacher_obs (agent 0), teacher_obs (agent 1)]
        # asker_obs_n = [asker_obs (agent 1), asker_obs (agent 0)]
        teacher_obs_n, asker_obs_n = get_meta_obs_n(
            student_n=student_n, 
            student_obs_n=student_obs_n,
            args=args)

        teacher_action_n = []
        asker_action_n = []
        for i_meta in range(2):
            i_asker = int(1 - i_meta)

            teacher_action_n.append(
                teacher_n[i_meta].select_stochastic_action(np.array(teacher_obs_n[i_meta]), total_step_count))
            asker_action_n.append(
                asker_n[i_asker].select_stochastic_action(np.array(asker_obs_n[i_meta]), total_step_count))

        advising_happened = False
        for i_meta, teacher_action, asker_action in zip([0, 1], teacher_action_n, asker_action_n):
            if asker_action > 0 and teacher_action[-1] > 0: 
                student_action_n[int(1 - i_meta)] = copy.deepcopy(teacher_action[0:2])
                advising_happened = True

        # Perform action
        new_env_obs_n, env_reward_n, env_done_n, _ = env.step(copy.deepcopy(student_action_n))
        if ep_timesteps + 1 == args.ep_max_timesteps:
            terminal = True

        # Add student memory
        new_student_obs_n = get_student_obs(new_env_obs_n, ep_timesteps + 1, args)
        student_reward_n = env_reward_n
        student_done = terminal if args.student_done else False

        for i_student, student in enumerate(student_n):
            # NOTE Order is student 0 and student 1
            student.add_memory(
                obs=student_obs_n,
                new_obs=new_student_obs_n,
                action=student_action_n,
                reward=student_reward_n,
                done=[float(student_done), float(student_done)])

        # Get next meta obs
        new_teacher_obs_n, new_asker_obs_n = get_meta_obs_n(
            student_n=student_n, 
            student_obs_n=new_student_obs_n,
            args=args)

        # update student policy
        for i_student, student in enumerate(student_n):
            student.update_policy(student_n, total_step_count)

        # Get meta reward
        meta_reward = get_teacher_reward(
            student_n, new_student_obs_n, advising_happened, log, tb_writer, args, total_step_count)

        # Add to meta memory
        meta_done = terminal if args.meta_done else False
        for i_meta, meta in enumerate([teacher_n[0], asker_n[1], teacher_n[1], asker_n[0]]):
            meta.add_memory(
                obs=[teacher_obs_n[0], asker_obs_n[0], teacher_obs_n[1], asker_obs_n[1]],
                new_obs=[new_teacher_obs_n[0], new_asker_obs_n[0], new_teacher_obs_n[1], new_asker_obs_n[1]],
                action=[teacher_action_n[0], asker_action_n[0], teacher_action_n[1], asker_action_n[1]],
                reward=[meta_reward, meta_reward, meta_reward, meta_reward],
                done=[float(meta_done), float(meta_done), float(meta_done), float(meta_done)])

        # For next timestep
        env_obs_n = new_env_obs_n
        ep_timesteps += 1
        total_step_count += 1
        session_step_count += 1
        ep_reward += env_reward_n[0]

        if terminal: 
            total_ep_count += 1
            session_ep_count += 1
            log[args.log_name].info("Train episode reward {} at episode {}".format(ep_reward, total_ep_count))
            tb_writer.add_scalar("reward/train_ep_reward", ep_reward, total_ep_count)

            return ep_reward


def train(student_n, teacher_n, asker_n, env, log, tb_writer, args):
    global session_step_count, session_ep_count

    while True:
        # Load trained students
        # This function will load one expert teacher and one new student
        # NOTE Both agents are trained
        load_trained_student_n(student_n)

        while True:
            if total_ep_count % 3 == 0:
                eval_progress(
                    env=env, 
                    student_n=student_n, 
                    log=log, 
                    tb_writer=tb_writer, 
                    args=args,
                    log_result=True)

            collect_one_traj(
                student_n=student_n,
                teacher_n=teacher_n,
                asker_n=asker_n,
                env=env,
                log=log,
                args=args, 
                tb_writer=tb_writer)

            # Teacher update
            if total_ep_count % 15 == 0:
                for meta in [teacher_n[0], asker_n[1], teacher_n[1], asker_n[0]]:
                    meta.update_policy(
                        agent_n=[teacher_n[0], asker_n[1], teacher_n[1], asker_n[0]], 
                        total_step_count=total_step_count)

            if total_ep_count % args.session == 0:
                session_step_count = 0
                session_ep_count = 0
                TRAIN_REWARD_N.clear()
                EVAL_REWARD_N.clear()
                break
