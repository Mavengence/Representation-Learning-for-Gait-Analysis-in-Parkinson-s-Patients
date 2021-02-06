import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from argparse import ArgumentParser
import os

parser = ArgumentParser()

from models.lstm.LSTM import ParkinsonLSTM
from models.resnet.ResNet import ResNet
from models.autoencoder_loss.autoencoder import ParkinsonResNetAutoEncoder
from models.autoencoder_sampler.autoencoder import ParkinsonAutoEncoder

from models.resnet.data import ParkinsonDataModuleResNet
from models.lstm.data import ParkinsonDataModuleLSTM
from models.autoencoder_loss.data import ParkinsonResNetDataModuleAE
from models.autoencoder_sampler.data import ParkinsonDataModuleAE


def main(args):
    dict_args = vars(args)
 
    if args.model_name == 'lstm':
        model = ParkinsonLSTM(**dict_args)
        datamodule = ParkinsonDataModuleLSTM(seq_len = dict_args['seq_len'], max_batch_len = dict_args['max_batch_len'])
        
    elif args.model_name == 'resnet':
        model = ResNet(max_batch_len = dict_args['max_batch_len'])
        datamodule = ParkinsonDataModuleResNet(max_batch_len = dict_args['max_batch_len'], batch_size=dict_args['batch_size'])
        
    elif args.model_name == 'autoencoder_loss':
        model = ParkinsonResNetAutoEncoder(**dict_args)
        datamodule = ParkinsonResNetDataModuleAE(max_batch_len = dict_args['max_batch_len'], batch_size=dict_args['batch_size'])
        
    elif args.model_name == 'autoencoder_sampler':
        model = ParkinsonAutoEncoder(**dict_args)
        datamodule = ParkinsonDataModuleAE(max_batch_len = dict_args['max_batch_len'])
    
    
    logger = TensorBoardLogger('tb_logs', name=dict_args["model_name"])
    
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
    
    current_trained_models = len(os.listdir("models/" + str(args.model_name) + "/checkpoints")) 
    model_save_path = './models/' + str(args.model_name) + '/checkpoints/training_' + str(current_trained_models)
    os.mkdir(model_save_path)
    
    model_checkpoint = ModelCheckpoint(save_top_k=1,
                                       dirpath=model_save_path,
                                       filename='./parkinson-{epoch:02d}-{val_loss:.2f}', #/parkinson-{epoch:02d}-{val_loss:.2f}
                                       monitor='val_loss',
                                       mode='min')
    
    trainer = Trainer.from_argparse_args(args, 
                                         #truncated_bptt_steps=2,
                                         auto_scale_batch_size=False,
                                         auto_lr_find=False, 
                                         fast_dev_run=False,
                                         callbacks=[early_stop_callback, model_checkpoint],
                                         logger=logger, 
                                         min_epochs=1,
                                         max_epochs=dict_args["epochs"])
    
    trainer.fit(model, datamodule)
    print("Model saved and training finished.")
    #trainer.test(model, datamodule)


def objective_optuna(trial):
    args = {
        "max_batch_len": 500,
        "epochs": 5000,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "network": "autoencoder"
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--model_name', type=str, default='resnet', help='lstm, resnet or autoencoder')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == 'lstm':
        parser = ParkinsonLSTM.add_model_specific_args(parser)
        
    elif temp_args.model_name == 'resnet':
        parser = ResNet.add_model_specific_args(parser)
    
    elif temp_args.model_name == 'autoencoder_sampler':
        parser = ParkinsonAutoEncoder.add_model_specific_args(parser)
        
    elif temp_args.model_name == 'autoencoder_loss':
        parser = ParkinsonResNetAutoEncoder.add_model_specific_args(parser)

    main(parser.parse_args())