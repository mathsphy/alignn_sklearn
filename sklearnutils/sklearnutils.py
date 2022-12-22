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
from alignn.graphs import Graph, StructureDataset,compute_bond_cosines
from jarvis.db.figshare import data
import pathlib
import dgl 

#%%
class GraphsToDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graphs, 
        line_graphs,
        ids,
        labels,
        classification=False
        ):
        self.graphs = graphs
        self.line_graphs = line_graphs
        self.ids = ids
        self.labels = labels
        
        self.labels = torch.tensor(self.labels).type(
            torch.get_default_dtype()
        )
        
        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)
            
        self.prepare_batch = alignn.graphs.prepare_line_graph_batch
        
    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        return self.graphs[idx], self.line_graphs[idx], self.labels[idx]

    @staticmethod
    def collate_line_graph(
        samples #: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.tensor(labels)




#%%
'''
need to think
first need to make sure that 
'''

def get_graphs(
        df: Union[pd.DataFrame,pd.Series],
        config: TrainingConfig,
        compute_line_graph=True
        ):
    import swifter
    atoms = df["atoms"].apply(Atoms.from_dict)
    print('Converting atoms object to crystal graphs')
    ''' 
    forces swifter to use dask to run parallel apply.
    '''
    # graphs = atoms.progress_apply(
    df_graphs = atoms.swifter.force_parallel().apply(
        lambda x: Graph.atom_dgl_multigraph(
            x,
            cutoff=config.cutoff,
            atom_features=config.atom_features,
            max_neighbors=config.max_neighbors,
            compute_line_graph=compute_line_graph,
            use_canonize=config.use_canonize,
        ))
    return df_graphs

def graph_to_line_graph(g):
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    return lg

def get_loader(
        df: Union[pd.DataFrame,pd.Series],
        config: TrainingConfig,
        drop_last: bool = True,
        shuffle: bool = True,
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

    '''
    determine the data type of X    
    '''
    X = dataset.iloc[:,0]
    var = X.iloc[0]
    if isinstance(var,dict):
        ''' Compute crystal graphs if the data type is atom dict '''
        graphs = get_graphs(dataset,config,compute_line_graph=False)
        torch_dataset = StructureDataset(
            dataset,
            graphs,
            target=config.target,
            atom_features=config.atom_features,
            line_graph=True,
            id_tag=config.id_tag,
            classification=config.classification_threshold is not None,
        )
        
    elif isinstance(var,dgl.DGLGraph):
        ''' var is crystal graph ''' 
        graphs = X 
        print('Computing line graphs')
        line_graphs = graphs.progress_apply(graph_to_line_graph)
        
        torch_dataset = GraphsToDataset(
            graphs = graphs,
            line_graphs = line_graphs,
            ids = dataset[config.id_tag],
            labels = dataset[config.target],
            classification=config.classification_threshold is not None,
            )
        
    elif (isinstance(var,tuple) and list(map(type, var)) == [dgl.DGLGraph, dgl.DGLGraph]):
        ''' var = (crystal graph, line graph) ''' 
        graphs = X.apply(lambda x: x[0])
        line_graphs = X.apply(lambda x: x[1])
        
        torch_dataset = GraphsToDataset(
            graphs = graphs,
            line_graphs = line_graphs,
            ids = dataset[config.id_tag],
            labels = dataset[config.target],
            classification=config.classification_threshold is not None,
            )
    
    else: 
        raise TypeError(f"Datatype of X not supported (datatype = {type(var)})") 
    
    loader = DataLoader(
        torch_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=torch_dataset.collate_line_graph,
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
    config=self.config
    # get df
    df = pd.concat([X,y], axis=1)
    
    # get train loader
    train_loader = get_loader(
            df = df,
            config=config,
            drop_last = True,
            shuffle = True,
            )
    # get trainer
    trainer = _get_trainer(self, train_loader)
    trainer.run(train_loader, max_epochs=config.epochs)


def _predict(
        self,
        X: Union[pd.DataFrame,pd.Series]
        ):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    config=self.config
    test_loader = get_loader(
            df = X,
            config=config,
            drop_last = False,
            shuffle = False,
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

    def fit(self, X, y):    
        _fit(self, X, y)
    
    def predict(self, X):
        y_pred = _predict(self, X)    
        return y_pred
    
    
#%%

if __name__ == "__main__":
    
    ''' An example usage of training a model on (1% of) the Jarvis dataset '''
    config_filename = 'config.json'
    config = loadjson(config_filename)
    config = TrainingConfig(**config)
    config.target = 'formation_energy_peratom'

    d = data('dft_3d') #choose a name of dataset from above
    df = pd.DataFrame(d).drop_duplicates('jid').set_index('jid')
    df = df.sample(frac=0.01,random_state=0)
    df['precomputed_graphs'] = get_graphs(df, config,compute_line_graph=True)

    
    model = AlignnLayerNorm(config)
    df2 = df.sample(frac=0.1,random_state=0)
    X = df2['precomputed_graphs']
    y = df2['formation_energy_peratom']
    # model.fit(X,y,precomputed_graphs=precomputed_graphs.loc[X.index])
    model.fit(X,y) 

    ids = set(df.index.tolist()) - set(df2.index.tolist())
    df1 = df.loc[list(ids)]
    X = df2['precomputed_graphs']
    y = df1['formation_energy_peratom']
    y_pred = model.predict(X)
    y_err = (y_pred - y).abs()
    
    
