import argparse

from model.model_trainer_crvae import Trainer
from utils import get_args

# set the parser
parser = argparse.ArgumentParser()
# set the hyperparameters from the parser
args = get_args(parser)
# create a trainer object
trainer = Trainer(args=args)
# train the model
trainer.train_loop()
