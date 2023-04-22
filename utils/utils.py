"""
@file utils.py

Utility functions across files
"""
import os
import json
import torch.nn as nn
import pytorch_lightning


def parse_args(parser):
    """
    Parse the cmd for a given configuration file and updates the arguments with its content
    As well, given the arguments, gets which model dynamics function is being used.
    :return: parsed arguments and model class
    """
    # Parse cmd line args
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Load in config file and update parser
    with open(parser.parse_args().config_path, 'rt') as f:
        args.__dict__.update(json.load(f))

    # Get the model type from args and add its specific arguments
    model_type = get_model(args.model)
    args.__dict__.update(model_type.get_model_specific_args())
    return args, model_type


def get_model(name):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    # Neural ODE
    if name == "node":
        from models.dynamics_models.NeuralODE import NeuralODE
        return NeuralODE

    # Given no correct model type, raise error
    raise NotImplementedError("Model type {} not implemented.".format(name))


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == "swish":
        return nn.SiLU()
    else:
        return None


def get_exp_versions(model, exptype):
    """ Return the version number for the latest lightning log and experiment num """
    # Set up lightning_logs if it doesn't exist
    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs/")

    if not os.path.exists("lightning_logs/version_0/"):
        os.mkdir("lightning_logs/version_0/")

    # Find version folder path
    top = 0
    for folder in os.listdir("lightning_logs/"):
        try:
            num = int(folder.split("_")[-1])
            top = num if num > top else top
        except ValueError:
            continue

    top += 1
    print("Version {}".format(top))

    # Set up paths if they don't exist
    if not os.path.exists("experiments/"):
        os.mkdir("experiments/")

    if not os.path.exists("experiments/{}".format(exptype)):
        os.mkdir("experiments/{}/".format(exptype))

    if not os.path.exists("experiments/{}/{}".format(exptype, model)):
        os.mkdir("experiments/{}/{}".format(exptype, model))

    # Find version folder path
    exptop = 0
    for folder in os.listdir("experiments/{}/{}/".format(exptype, model)):
        try:
            num = int(folder.split("_")[-1])
            exptop = num if num > exptop else exptop
        except ValueError:
            continue

    exptop += 1
    print("Exp Top {}".format(exptop))
    return top, exptop


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1', 'True',  'T',  'true'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0', 'False', 'F', 'false'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
