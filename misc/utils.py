import logging
import git
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    # Log arguments
    for (name, value) in vars(args).items():
        log[args.log_name].info("{}: {}".format(name, value))

    return log


def check_github(path, branch_name):
    """Checks whether the path has a correct branch name"""
    repo = git.Repo(path)
    branch = repo.active_branch
    assert branch.name == branch_name, "Branch name does not equal the desired branch"


def make_env(args):
    """Load multi-agent particle environment
    This code is modified from: https://github.com/openai/maddpg/blob/master/experiments/train.py
    """
    # Check github branch
    check_github(
        path="./thirdparty/multiagent-particle-envs",
        branch_name="opponent")

    # Load multi-agent particle env
    scenario = scenarios.load(args.env_name + ".py").Scenario()
    world = scenario.make_world()
    done_callback = None

    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=done_callback)

    assert env.discrete_action_space is False, "For cont. action, this flag must be False"

    return env


def normalize(value, min_value, max_value):
    assert min_value < max_value
    return 2. * (value - min_value) / float(max_value - min_value) - 1.
