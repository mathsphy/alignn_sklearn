#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Create a sklearn interface

"""

from typing import Any, Dict, Union
from torch import nn
import pandas as pd
import os
import ignite
import torch

import alignn
from alignn.config import TrainingConfig
from alignn.train import group_decay,setup_optimizer

from jarvis.core.atoms import Atoms

from ignite.metrics import Loss, MeanAbsoluteError, RootMeanSquaredError
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)

from alignn.data import get_torch_dataset, load_graphs
from torch.utils.data import DataLoader
    
from alignn.train import (
    thresholded_output_transform, 
    activated_output_transform
    )

from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from tqdm import tqdm

try:
    from ignite.contrib.handlers.stores import EpochOutputStore
    # For different version of pytorch-ignite
except Exception:
    from ignite.handlers.stores import EpochOutputStore
from jarvis.db.jsonutils import dumpjson
# from ignite.handlers import EarlyStopping
# from ignite.contrib.handlers import TensorboardLogger,global_step_from_engine

from jarvis.db.jsonutils import loadjson
from alignn.graphs import Graph, StructureDataset
from jarvis.db.figshare import data
import pathlib

#%%

def poscars2df(wd: str, target: str = 'target') -> pd.DataFrame:
    '''
    convert a folder of POSCARs and id_props.csv into a df with two columns:
        one called 'atoms' containing dict representations of jarvis.core.atoms.Atoms objects
        other called 'y' containing labels   
        
    The filenames of POSCARs are given as the 1st column in id_props.csv, and 
    will be used as index of df.
    '''

    id_tag = 'jid'
    
    df = pd.read_csv(wd+'/id_props.csv') 
    df.columns = [id_tag,target]     
    df['atoms'] = df[id_tag].apply(
        lambda x: Atoms.from_poscar(wd+'/'+x).to_dict()
        )
    df.set_index(id_tag,drop=False)
    return df 



#%%

def get_graphs(
        df: Union[pd.DataFrame,pd.Series],
        config: TrainingConfig,
        ):
    import swifter
    atoms = df["atoms"].apply(Atoms.from_dict)
    print('Converting atoms object to crystal graphs')
    ''' 
    forces swifter to use dask to run parallel apply.
    '''
    # graphs = atoms.progress_apply(
    graphs = atoms.swifter.force_parallel().apply(
        lambda x: Graph.atom_dgl_multigraph(
            x,
            cutoff=config.cutoff,
            atom_features="atomic_number",
            max_neighbors=config.max_neighbors,
            compute_line_graph=False,
            use_canonize=config.use_canonize,
        ))
    return graphs

# def get_graphs(
#         df: Union[pd.DataFrame,pd.Series],
#         config: TrainingConfig,
#         ):
#     graphs = load_graphs(
#         df,
#         neighbor_strategy=config.neighbor_strategy,
#         use_canonize=config.use_canonize,
#         cutoff=config.cutoff,
#         max_neighbors=config.max_neighbors,
#     )
#     return graphs


def get_loader(
        df: Union[pd.DataFrame,pd.Series],
        config: TrainingConfig,
        drop_last: bool = True,
        shuffle: bool = True,
        precomputed_graphs = None
                 ):

    dataset = df 
    '''
    if 1d array, then dataset contains only `X` (used in the predict method), 
    in this case, add an additional target column for compatibility
    '''
    if isinstance(dataset, pd.Series):
        dataset = dataset.to_frame()
        dataset[config.target] = -9999 # place holder for the predict method
    elif isinstance(dataset, pd.DataFrame):
        if dataset.shape[1] == 1:
            dataset[config.target] = -9999 # place holder for the predict method
    ''' Add a column for id_tag '''
    dataset[config.id_tag] = dataset.index.tolist()

    
    if precomputed_graphs is not None: 
        graphs = precomputed_graphs
    else:
        graphs = get_graphs(dataset,config)
    
    torch_dataset = StructureDataset(
        dataset,
        graphs,
        target=config.target,
        # target_atomwise=target_atomwise,
        # target_grad=target_grad,
        # target_stress=target_stress,
        atom_features=config.atom_features,
        line_graph=True,
        id_tag=config.id_tag,
        classification=config.classification_threshold is not None,
    )
    
    collate_fn = torch_dataset.collate_line_graph    
    loader = DataLoader(
        torch_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )  
    return loader

def _init(self, config: TrainingConfig, chk_file):         
    self.config = config        
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the checkpoint file
    if chk_file is not None:       
        self.load_state_dict(
            torch.load(chk_file, map_location=self.device)["model"]
            )
        print(f'Checkpoint file {chk_file} loaded')
    self.to(self.device)
    
def _fit(
        self, 
        X: Union[pd.DataFrame,pd.Series], 
        y: Union[pd.DataFrame,pd.Series],
        precomputed_graphs
        ):
    '''
    Parameters
    ----------
    X : Union[pd.DataFrame,pd.Series]
        A column of Atoms.to_dict()
    y : Union[pd.DataFrame,pd.Series]
        A column of values

    Returns
    -------
    None.

    '''
    
    # get df
    df = pd.concat([X,y], axis=1)
    
    # get train loader
    train_loader = get_loader(
            df = df,
            config=self.config,
            drop_last = True,
            shuffle = True,
            precomputed_graphs = precomputed_graphs
            )
    # get trainer
    trainer = _get_trainer(self, train_loader)
    trainer.run(train_loader, max_epochs=config.epochs)


def _predict(
        self,
        X: Union[pd.DataFrame,pd.Series],
        precomputed_graphs
        ):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    test_loader = get_loader(
            df = X,
            config=self.config,
            drop_last = False,
            shuffle = False,
            precomputed_graphs = precomputed_graphs
            )        
    col_ids = []
    col_pred = []
    with torch.no_grad():
        ids_chunks = list(chunks(
            test_loader.dataset.ids,
            test_loader.batch_size
            ))
        for dat, ids in tqdm(zip(test_loader, ids_chunks)):
            g, lg, target = dat
            out_data = self([g.to(self.device), lg.to(self.device)])
            out_data = out_data.cpu().numpy().tolist()
            col_ids.extend(ids)
            col_pred.extend(out_data)
    results = pd.Series(data=col_pred,index=col_ids,name=config.target)
    return results
    

def _get_trainer(self, train_loader):

    config = self.config
    
    ''' 
    set up scheduler 
    ''' 
    
    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(self)
    optimizer = setup_optimizer(params, config)
    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
        
    elif config.scheduler == "step":
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )
        
    '''
    select configured loss function
    '''
    
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
        "zig": alignn.models.modified_cgcnn.ZeroInflatedGammaLoss(),
        }
    criterion = criteria[config.criterion] 
    
    ''' 
    set up criterion and metrics
    '''
    
    metrics = {
        "loss": Loss(criterion), 
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError()
        }
    
    output_transform = alignn.train.make_standard_scalar_and_pca
    
    if config.model.output_features > 1 and config.standard_scalar_and_pca:
        # metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
        metrics = {
            "loss": Loss(
                criterion, output_transform=output_transform
            ),
            "mae": MeanAbsoluteError(
                output_transform=output_transform
            ),
        }
        
    if config.criterion == "zig":

        def zig_prediction_transform(x):
            output, y = x
            return criterion.predict(output), y

        metrics = {
            "loss": Loss(criterion),
            "mae": MeanAbsoluteError(
                output_transform=zig_prediction_transform
            ),
        }

    if config.classification_threshold is not None:
        criterion = nn.NLLLoss()

        metrics = {
            "accuracy": Accuracy(
                output_transform=thresholded_output_transform
            ),
            "precision": Precision(
                output_transform=thresholded_output_transform
            ),
            "recall": Recall(output_transform=thresholded_output_transform),
            "rocauc": ROC_AUC(output_transform=activated_output_transform),
            "roccurve": RocCurve(output_transform=activated_output_transform),
            "confmat": ConfusionMatrix(
                output_transform=thresholded_output_transform, num_classes=2
            ),
        }
        
    '''
    Set up training engine and evaluators 
    '''
    
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)    
    else:
        deterministic = False
    
    prepare_batch = train_loader.dataset.prepare_batch
    trainer = create_supervised_trainer(
        self,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=self.device,
        deterministic = deterministic,
        # output_transform=make_standard_scalar_and_pca,
    )
    
    ''' 
    Set up various event handlers for the trainer
    '''
    
    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
        
    # add the "writing checkpoint file" event handler
    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": self,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(config.output_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
        
    # attach progress bar to the trainer
    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})


    '''
    log performance
    '''
    
    train_evaluator = create_supervised_evaluator(
        self,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=self.device,
        # output_transform=make_standard_scalar_and_pca,
    )

    history = {
        "train": {m: [] for m in metrics.keys()},
        # "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)
    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)

        tmetrics = train_evaluator.state.metrics
        for metric in metrics.keys():
            tm = tmetrics[metric]
            if metric == "roccurve":
                tm = [k.tolist() for k in tm]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()

            history["train"][metric].append(tm)

        if config.store_outputs:
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            if config.classification_threshold is None:
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}, Train_RMSE: {tmetrics['rmse']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
    return trainer

#%%
   
class AlignnBatchNorm(alignn.models.alignn.ALIGNN):

    def __init__(self, config: TrainingConfig, chk_file=None):        
        super().__init__(config.model)   
        _init(self, config, chk_file)

    def fit(self, X, y, precomputed_graphs=None):    
        _fit(self, X, y, precomputed_graphs)
    
    def predict(self, X, precomputed_graphs=None):
        y_pred = _predict(self, X, precomputed_graphs)    
        return y_pred


class AlignnLayerNorm(alignn.models.alignn_layernorm.ALIGNN):

    def __init__(self, config: TrainingConfig, chk_file=None):        
        super().__init__(config.model)   
        _init(self, config, chk_file)

    def fit(self, X, y, precomputed_graphs=None):    
        _fit(self, X, y, precomputed_graphs)
    
    def predict(self, X, precomputed_graphs=None):
        y_pred = _predict(self, X, precomputed_graphs)    
        return y_pred
    
    
#%%
if __name__ == "__main__":
    
    ''' An example usage of training a model on (10% of) the Jarvis dataset '''
    config_filename = 'config.json'
    config = loadjson(config_filename)
    config = TrainingConfig(**config)
    config.target = 'formation_energy_peratom'

    pkl_file = 'jarvis.pkl'
    if pathlib.Path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
        precomputed_graphs = df['precomputed_graphs']
    else:
        d = data('dft_3d') #choose a name of dataset from above
        df = pd.DataFrame(d).drop_duplicates('jid').set_index('jid')
        df = df.sample(frac=1,random_state=0)
        precomputed_graphs = get_graphs(df, config)
        df['precomputed_graphs'] = precomputed_graphs
        df.to_pickle()

    model = AlignnLayerNorm(config)
    
    df2 = df.sample(frac=0.8,random_state=0)
    X = df2['atoms']
    y = df2['formation_energy_peratom']
    model.fit(X,y,precomputed_graphs=precomputed_graphs.loc[X.index])
    
    ids = set(df.index.tolist()) - set(df2.index.tolist())
    df1 = df.loc[list(ids)]
    X = df1['atoms']
    y = df1['formation_energy_peratom']
    y_pred = model.predict(X,precomputed_graphs=precomputed_graphs.loc[X.index])
    y_err = (y_pred - y).abs()
