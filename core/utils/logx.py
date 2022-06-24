"""
Some simple logging functionality, inspired by rllab's logging.
Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

ref: spinup.utils.logx
ref: spinup.utils.serialization_util
"""
import json
import joblib
import shutil
import numpy as np
import tensorflow as tf
import torch
import os.path as osp, time, atexit, os
import warnings
import json

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def restore_tf_graph(sess, fpath):
    """
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs' 
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``. 
    """
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], fpath)
    model_info = joblib.load(osp.join(fpath, "model_info.pkl"))
    graph = tf.get_default_graph()
    model = dict()
    model.update(
        {k: graph.get_tensor_by_name(v) for k, v in model_info["inputs"].items()}
    )
    model.update(
        {k: graph.get_tensor_by_name(v) for k, v in model_info["outputs"].items()}
    )
    return model


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.output_dir = output_dir or "experiments/%i" % int(time.time())
        if osp.exists(self.output_dir):
            print(
                "Warning: Log dir %s already exists! Storing info there anyway."
                % self.output_dir
            )
        else:
            os.makedirs(self.output_dir)
        
        self.output_fname = output_fname
        self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def _create_file(self):
        if self.output_file is not None:
            return
        self.output_file = open(osp.join(self.output_dir, self.output_fname), "w")
        atexit.register(self.output_file.close)
        print(
            colorize("Logging data to %s" % self.output_file.name, "green", bold=True)
        )

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that you didn't include in the first iteration"
                % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
            % key
        )
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json["exp_name"] = self.exp_name
        output = json.dumps(
            config_json, separators=(",", ":\t"), indent=4, sort_keys=True
        )
        print(colorize("Saving config:\n", color="cyan", bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), "w") as out:
            out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        fname = "vars.pkl" if itr is None else "vars%d.pkl" % itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, fname))
        except:
            self.log("Warning: could not pickle state_dict.", color="red")
        if hasattr(self, "tf_saver_elements"):
            self._tf_simple_save(itr)
        if hasattr(self, "pytorch_saver_elements"):
            self._pytorch_simple_save(itr)

    def setup_tf_saver(self, sess, inputs, outputs):
        """
        Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the 
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        self.tf_saver_info = {
            "inputs": {k: v.name for k, v in inputs.items()},
            "outputs": {k: v.name for k, v in outputs.items()},
        }

    def _tf_simple_save(self, itr=None):
        """
        Uses simple_save to save a trained model, plus info to make it easy
        to associated tensors to variables after restore. 
        """
        assert hasattr(
            self, "tf_saver_elements"
        ), "First have to setup saving with self.setup_tf_saver"
        fpath = "tf1_save" + ("%d" % itr if itr is not None else "")
        fpath = osp.join(self.output_dir, fpath)
        if osp.exists(fpath):
            # simple_save refuses to be useful if fpath already exists,
            # so just delete fpath if it's there.
            shutil.rmtree(fpath)
        tf.saved_model.simple_save(export_dir=fpath, **self.tf_saver_elements)
        joblib.dump(self.tf_saver_info, osp.join(fpath, "model_info.pkl"))

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        assert hasattr(
            self, "pytorch_saver_elements"
        ), "First have to setup saving with self.setup_pytorch_saver"
        fpath = "pyt_save"
        fpath = osp.join(self.output_dir, fpath)
        fname = "model" + ("%d" % itr if itr is not None else "") + ".pt"
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We are using a non-recommended way of saving PyTorch models,
            # by pickling whole objects (which are dependent on the exact
            # directory structure at the time of saving) as opposed to
            # just saving network weights. This works sufficiently well
            # for the purposes of Spinning Up, but you may want to do
            # something different for your personal PyTorch project.
            # We use a catch_warnings() context to avoid the warnings about
            # not being able to save the source code.
            torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        self._create_file()
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def reset(self):
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
