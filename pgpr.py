from train_save.task import Task

class DataLoader(Task):
    def __init__(self, name: str, path: str):
        super().__init__(name=name, path=path)

    def go(self):
        