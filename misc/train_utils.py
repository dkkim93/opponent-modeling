import numpy as np
import torch
from misc.utils import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_VALUE_CLIP_MIN = -50.
Q_VALUE_CLIP_MAX = 0.
AVG_CLIP_MIN = -50.
AVG_CLIP_MAX = 0.


########################################################################################
# MISC
def load_trained_student_n(student_n):
    # Randomly sample two students
    student_n[0].load_model(
        filename="student0_reward-37.393_seed7_ep3500",
        directory="./pytorch_models/knowledge_transfer/push_two/primitive/mode0")

    student_n[1].load_model(
        filename="student0_reward-40.300_seed3_ep5000",
        directory="./pytorch_models/knowledge_transfer/push_two/primitive/mode1")


def get_q_value_n(agent_n, agent_obs_n, agent_action_n):
    assert len(agent_action_n) == 2

    agent_obs_n = np.concatenate([agent_obs_n[0], agent_obs_n[1]]).reshape(1, -1)
    agent_obs_n = torch.FloatTensor(agent_obs_n).to(device)

    agent_action_n = np.concatenate([agent_action_n[0], agent_action_n[1]]).reshape(1, -1)
    agent_action_n = torch.FloatTensor(agent_action_n).to(device)

    q_value_n = []
    for agent in agent_n:
        q_value = agent.policy.critic.Q1(agent_obs_n, agent_action_n)
        q_value = q_value.cpu().data.numpy().flatten()[0]
        q_value_n.append(q_value)

    return q_value_n


def get_avg_reward(TRAIN_REWARD_N, EVAL_REWARD_N, args):
    if len(TRAIN_REWARD_N) == 0:
        avg_train_reward = AVG_CLIP_MIN  # NOTE Initialize to be min_value
    else:
        if len(TRAIN_REWARD_N) > args.window_size:
            avg_train_reward = np.average(TRAIN_REWARD_N[-args.window_size:-1])
        else:
            avg_train_reward = np.average(TRAIN_REWARD_N)
    avg_train_reward = np.clip(avg_train_reward, a_min=AVG_CLIP_MIN, a_max=AVG_CLIP_MAX)
    avg_train_reward = normalize(avg_train_reward, min_value=AVG_CLIP_MIN, max_value=AVG_CLIP_MAX)

    if len(EVAL_REWARD_N) == 0:
        raise ValueError()
    else:
        if len(EVAL_REWARD_N) > args.window_size:
            avg_eval_reward = np.average(EVAL_REWARD_N[-args.window_size:-1])
        else:
            avg_eval_reward = np.average(EVAL_REWARD_N)
    avg_eval_reward = np.clip(avg_eval_reward, a_min=AVG_CLIP_MIN, a_max=AVG_CLIP_MAX)
    avg_eval_reward = normalize(avg_eval_reward, min_value=AVG_CLIP_MIN, max_value=AVG_CLIP_MAX)

    return avg_train_reward, avg_eval_reward


########################################################################################
# STUDENT
def get_student_obs(env_obs_n, ep_timesteps, args):
    student_obs_n = []

    if args.student_done:
        remaining_timesteps = normalize(
            value=(args.ep_max_timesteps - ep_timesteps), 
            min_value=0.,
            max_value=float(args.ep_max_timesteps))
        remaining_timesteps = np.array([remaining_timesteps])
        
        for env_obs in env_obs_n:
            student_obs = np.concatenate([env_obs, remaining_timesteps])
            student_obs_n.append(student_obs)

        return student_obs_n
    else:
        return env_obs_n


########################################################################################
# TEACHER
def concat_in_order_teacher(obs):
    return np.concatenate([
        obs["student_obs_n"][0].flatten(), 
        obs["student_obs_n"][1].flatten(), 
        np.asarray(obs["random_action_n"]).flatten(),
        np.asarray(obs["student_agent_q_values"]).flatten(),
        np.asarray(obs["teacher_agent_q_values"]).flatten()])


def concat_in_order_asker(obs):
    return np.concatenate([
        obs["student_obs_n"][0].flatten(), 
        obs["student_obs_n"][1].flatten(), 
        np.asarray(obs["random_action_n"]).flatten(),
        np.asarray(obs["student_agent_q_values"]).flatten()])


def meta_input_process(student_n, student_obs_n, i_teacher, i_student, args):
    obs = {}
    obs["student_obs_n"] = student_obs_n  # Task observation

    # Sample random action
    obs["random_action_n"] = []
    for _ in range(15):
        random_action_n = [
            np.random.uniform(low=-1, high=1., size=2), 
            np.random.uniform(low=-1, high=1., size=2)]
        obs["random_action_n"].append(random_action_n)

    # Get Q-values using student critic
    obs["student_agent_q_values"] = []
    obs["teacher_agent_q_values"] = []
    for i_action in range(15):
        q_value_n = get_q_value_n(student_n, student_obs_n, obs["random_action_n"][i_action])
        q_value_n = np.clip(q_value_n, a_min=Q_VALUE_CLIP_MIN, a_max=Q_VALUE_CLIP_MAX)

        q_value_student = np.array([normalize(
            value=q_value_n[i_student],
            min_value=Q_VALUE_CLIP_MIN,
            max_value=Q_VALUE_CLIP_MAX)])
        obs["student_agent_q_values"].append(q_value_student)

        q_value_teacher = np.array([normalize(
            value=q_value_n[i_teacher],
            min_value=Q_VALUE_CLIP_MIN,
            max_value=Q_VALUE_CLIP_MAX)])
        obs["teacher_agent_q_values"].append(q_value_teacher)

    return concat_in_order_teacher(obs), concat_in_order_asker(obs)


def get_meta_obs_n(student_n, student_obs_n, args):
    teacher_obs_n = []
    asker_obs_n = []
    for i_teacher in range(2):
        i_asker = int(1 - i_teacher)
        assert (i_teacher + i_asker) == 1

        teacher_obs, asker_obs = meta_input_process(
            student_n=student_n, 
            student_obs_n=student_obs_n, 
            i_teacher=i_teacher, 
            i_student=i_asker,
            args=args)

        teacher_obs_n.append(teacher_obs)
        asker_obs_n.append(asker_obs)

    return teacher_obs_n, asker_obs_n


def get_student_value(student_n, student_obs_n, i_student):
    student_action_n = []
    for student, student_obs in zip(student_n, student_obs_n):
        student_action = student.select_deterministic_action(np.array(student_obs))
        student_action_n.append(student_action)

    q_value_n = []
    q_value_n.append(get_q_value_n(student_n, student_obs_n, student_action_n)[i_student])

    for _ in range(15):
        random_action_n = [
            np.random.uniform(low=-1, high=1., size=2), 
            np.random.uniform(low=-1, high=1., size=2)]
        q_value_n.append(get_q_value_n(student_n, student_obs_n, random_action_n)[i_student])

    value = np.max(q_value_n)
    return value


def get_teacher_reward(student_n, new_student_obs_n, advising_happend, log, tb_writer, args, total_step_count):
    value_n = []
    for i_teacher in range(2):
        i_student = int(1 - i_teacher)
        value = get_student_value(student_n, new_student_obs_n, i_student)
        tb_writer.add_scalar("debug/value_" + str(i_student), value, total_step_count)

        value_n.append(value)

    teacher_reward_n = 0.
    for i_teacher in range(2):
        if advising_happend is False:
            teacher_reward_n += 0.
        else:
            threshold = -13.
            if value_n[i_teacher] > threshold:
                teacher_reward_n += 1.
            else:
                teacher_reward_n += 0.

    tb_writer.add_scalar("reward/Teacher_Reward", teacher_reward_n, total_step_count)

    return teacher_reward_n


def save(agent_n, total_eps):
    for agent in agent_n:
        agent.save_weight(
            filename=agent.name + "_" + str(total_eps),
            directory="./pytorch_models")
