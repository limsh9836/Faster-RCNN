import os
import importlib

for module in os.listdir(os.path.dirname(__file__)):
    if module.endswith('.py') and module != "__init__.py":
        importlib.import_module('.{}'.format(module.strip(".py")), package=__name__)