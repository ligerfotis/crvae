import argparse

from model.model_trainer_crvae import CRVAE_Trainer
from model.model_trainer_vae import VAE_Trainer
from utils import get_args

# set the parser
parser = argparse.ArgumentParser()
# set the hyperparameters from the parser
args = get_args(parser)
# create a trainer object
if args.ae_type == "crvae":
    trainer = CRVAE_Trainer(args=args)
elif args.ae_type == "vae":
    trainer = VAE_Trainer(args=args)
else:
    raise Exception("unknown type of model in 'train.py'")
# train the model
trainer.train_loop()
