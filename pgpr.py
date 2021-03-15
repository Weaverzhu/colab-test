from train_save.task import Task
from PGPR.train_agent import ActorCritic, ACDataLoader
from PGPR.kg_env import BatchKGEnvironment


import yaml


class Train(Task):
    def __init__(self, model_params, data_params):
        self.model = ActorCritic(**model_params)
        self.data_loader = ActorCritic(**data_params)

    def go(self):
        pass
