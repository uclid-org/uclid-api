from uclid.builder import UclidModule


class Module:
    def __init__(self, name):
        self.module = UclidModule(name)

    def __str__(self):
        return self.module.__inject__()
