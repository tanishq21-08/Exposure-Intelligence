import yaml
import os


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")



with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

