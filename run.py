# change working directory
import os
import sys
workdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(workdir)
sys.path.insert(0, workdir)

import argparse
from core.exps import EXPS
from pathlib import Path
from omegaconf import OmegaConf
from pydoc import locate
from ast import literal_eval
from inspect import isclass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-tr", "--trainer", type=str, help="path to trainer module")
    parser.add_argument("-cfg", "--config", type=Path, help="path to configuration file", required=True)
    args, unknown = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    for kv in unknown:
        assert "=" in kv
        key, val = kv.split("=", 1)
        try:
            val = literal_eval(val)
        except:
            pass
        OmegaConf.update(config, key, val, merge=False)

    if os.environ.get("CLUSTER") == "mist":
        if config.experiment:
            config.experiment.root_dir = os.path.join(os.environ["SCRATCH"], config.experiment.root_dir)

    trainer = EXPS[config.experiment.engine]
    if isclass(trainer):
        trainer(config).train()
    else: # backward compatibility
        trainer(config)
