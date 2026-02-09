import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

config = {
    'logs': 'logs/'
}

# Resolve Data Directory (project-local only)
DATA_DIR = os.path.join(base_dir, 'datasets')

datasets_path = {
    'la': os.path.join(DATA_DIR, 'metr_la'),
    'bay': os.path.join(DATA_DIR, 'pems_bay'),
    'synthetic': os.path.join(DATA_DIR, 'synthetic'),
    'pems': os.path.join(DATA_DIR, 'pems'),
    'beijing12': os.path.join(DATA_DIR, 'beijing12'),
    'vessel': os.path.join(DATA_DIR, 'vessel_ais'),
}
epsilon = 1e-8

for k, v in config.items():
    config[k] = os.path.join(base_dir, v)