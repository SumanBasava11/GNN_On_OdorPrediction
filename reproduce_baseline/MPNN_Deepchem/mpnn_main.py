import os
import json
import torch
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import deepchem as dc
from datetime import datetime
from reproduce_baseline.MPNN_Deepchem.mpnn_pom_deepchem import *
from reproduce_baseline.MPNN_Deepchem.GraphFeaturizer_deepchem import GraphFeaturizer, GraphConvConstants
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

#check for the precision calculation
# from where the y_pred is coming?

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

# Config
DATASET = 'C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv'
SMILES_FIELD = 'smiles'

def get_tasks_from_csv(path):
    df = pd.read_csv(path, nrows=1)
    return df.columns[2:].tolist()

TASKS = get_tasks_from_csv(DATASET)

class CV:
    def __init__(self, model_builder, n_folds, device=None):
        self.model_builder = model_builder
        self.n_folds = n_folds
        self.device = device

    def generate_folds(self, dataset):
        splitter = dc.splits.RandomStratifiedSplitter()
        self.folds_list = splitter.k_fold_split(dataset=dataset, k=self.n_folds)
        return self.folds_list

    def cross_validation(self, model_params, logdir=None, max_epoch=10, save_best_ckpt=False):
        all_train_auc, all_val_auc = [], []
        all_train_prec, all_val_prec = [], []
        all_train_recall, all_val_recall = [], []
        all_train_f1, all_val_f1 = [], []

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode='classification')

        def make_metric(metric_func, name):
            def metric_wrapper(y_true, y_pred):
                y_true_bin = y_true.astype(int)
                y_pred_bin = (y_pred > 0.5).astype(int)

                return metric_func(y_true_bin, y_pred_bin, average='macro')

            return dc.metrics.Metric(
                metric_wrapper,
                name=name,
                mode='classification',
                classification_handling_mode='direct'
            )
        # def make_metric(metric_func, name):
        #     return dc.metrics.Metric(
        #         lambda y, y_pred: metric_func(y, (y_pred > 0.5).astype(int), average='macro'),
        #         name=name, mode='classification',
        #         classification_handling_mode='direct'
        #     )

        precision_metric = make_metric(precision_score, 'precision_score')
        recall_metric = make_metric(recall_score, 'recall_score')
        f1_metric = make_metric(f1_score, 'f1_score')

        for fold_num, (train_dataset, valid_dataset) in enumerate(self.folds_list):
            logger.info(f"Training Fold {fold_num + 1}/{self.n_folds}")
            model_dir = os.path.join(logdir, f"fold_{fold_num + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(model_dir, exist_ok=True)

            model_params['model_dir'] = model_dir
            model_params['class_imbalance_ratio'] = get_class_imbalance_ratio(train_dataset)
           
            if self.device:
                model_params['device_name'] = self.device

            model = self.model_builder(**model_params)
            print(model.model)

            best_val_auc = 0
            for epoch in tqdm(range(1, max_epoch + 1)):
                model.fit(train_dataset, nb_epoch=1, max_checkpoints_to_keep=1, deterministic=False, restore=epoch > 1)

                # Evaluate all metrics at once
                train_metrics = model.evaluate(train_dataset, [metric, precision_metric, recall_metric, f1_metric])
                val_metrics = model.evaluate(valid_dataset, [metric, precision_metric, recall_metric, f1_metric])

                train_auc = train_metrics['roc_auc_score']
                val_auc = val_metrics['roc_auc_score']
                train_prec = train_metrics['precision_score']
                val_prec = val_metrics['precision_score']
                train_rec = train_metrics['recall_score']
                val_rec = val_metrics['recall_score']
                train_f1 = train_metrics['f1_score']
                val_f1 = val_metrics['f1_score']

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_metrics = (train_auc, val_auc, train_prec, val_prec, train_rec, val_rec, train_f1, val_f1)
                    if save_best_ckpt:
                        torch.save(model.model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

                logger.info(
                    f"[Epoch {epoch}] "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f} | "
                    f"Train Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f} | "
                    f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}"
                )

            (ta, va, tp, vp, tr, vr, tf, vf) = best_metrics
            all_train_auc.append(ta); all_val_auc.append(va)
            all_train_prec.append(tp); all_val_prec.append(vp)
            all_train_recall.append(tr); all_val_recall.append(vr)
            all_train_f1.append(tf); all_val_f1.append(vf)
            
            del model
            torch.cuda.empty_cache()

        print("\n=== Mean Metrics Across All Folds ===")
        print(f"Train ROC AUC:       {np.mean(all_train_auc):.4f}")
        print(f"Validation ROC AUC:  {np.mean(all_val_auc):.4f}")
        print(f"Train Precision:     {np.mean(all_train_prec):.4f}")
        print(f"Validation Precision:{np.mean(all_val_prec):.4f}")
        print(f"Train Recall:        {np.mean(all_train_recall):.4f}")
        print(f"Validation Recall:   {np.mean(all_val_recall):.4f}")
        print(f"Train F1 Score:      {np.mean(all_train_f1):.4f}")
        print(f"Validation F1 Score: {np.mean(all_val_f1):.4f}")

        return np.mean(all_train_auc), np.mean(all_val_auc)

def main():
    featurizer = GraphFeaturizer()
    loader = dc.data.CSVLoader(tasks=TASKS, feature_field=SMILES_FIELD, featurizer=featurizer)
    dataset = loader.create_dataset(inputs=[DATASET])
    n_tasks = len(dataset.tasks)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_builder = lambda **params: MPNNPOMModel(
        n_tasks=n_tasks,
        mode='classification',
        number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1,
        **params
    )

    model_params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'dropout': 0.2,
        'dense_layer_size': 128,
        'number_of_molecules': 1,
        'model_dir': './models/fixed_config'
    }

    cv = CV(model_builder=model_builder, n_folds=5, device=device)
    cv.generate_folds(dataset)
    os.makedirs(model_params['model_dir'], exist_ok=True)

    train_score, val_score = cv.cross_validation(
        model_params=model_params,
        logdir=model_params['model_dir'],
        max_epoch=10,
        save_best_ckpt=True
    )

    logger.info(f"Final Train AUROC: {train_score:.4f}")
    logger.info(f"Final Val AUROC:   {val_score:.4f}")

if __name__ == "__main__":
    main()
