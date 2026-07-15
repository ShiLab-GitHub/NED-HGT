import os
import math
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import os
# os.environ['WANDB_MODE'] = 'disabled'  # Disable wandb
import wandb
import dill
from data import create_dataloader
from model import NedHGT as Model
from schedular import NoamLR
from utils import get_func,remove_nan_label

from torch.utils.tensorboard import SummaryWriter

def ci(y,f):
    y = y.squeeze().numpy()
    f = f.squeeze().numpy()
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def mae(y, f):
    y = y.squeeze().numpy()
    f = f.squeeze().numpy()
    return np.mean(np.abs(y - f))

def r2(y, f):
    y = y.squeeze().numpy()
    f = f.squeeze().numpy()
    ss_res = np.sum((y - f) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def evaluate(dataloader,model,device,metric_fn,metric_dtype,task):
    metric = 0
    ci_value_sum = 0
    mae_value_sum = 0
    r2_value_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for bg,labels,gf in dataloader:
            bg,labels,gf = bg.to(device),labels.type(metric_dtype),gf.to(device)
            pred = model(bg,gf).cpu().detach()
            if task == 'classification':
                pred = torch.sigmoid(pred)
            elif task == 'multiclass':
                pred = torch.softmax(pred,dim=1)
            num_task =  pred.size(1)
            if num_task >1:
                m = 0
                ci_value = 0
                mae_value = 0
                r2_value = 0
                for i in range(num_task):
                    try:
                        m += metric_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                        ci_value += ci(*remove_nan_label(labels[:,i],pred[:,i]))
                        mae_value += mae(*remove_nan_label(labels[:,i],pred[:,i]))
                        r2_value += r2(*remove_nan_label(labels[:,i],pred[:,i]))
                    except:
                        print(f'only one class for task {i}')
                m = m/num_task
                ci_value = ci_value/num_task
                mae_value = mae_value/num_task
                r2_value = r2_value/num_task
            else:
                m = metric_fn(pred,labels.reshape(pred.shape))
                ci_value = ci(labels.reshape(pred.shape),pred)
                mae_value = mae(labels.reshape(pred.shape),pred)
                r2_value = r2(labels.reshape(pred.shape),pred)
            metric += m.item()*len(labels)
            ci_value_sum += ci_value*len(labels)
            mae_value_sum += mae_value*len(labels)
            r2_value_sum += r2_value*len(labels)
            total_samples += len(labels)
    
    metric = metric/total_samples
    ci_value_sum = ci_value_sum/total_samples
    mae_value_sum = mae_value_sum/total_samples
    r2_value_sum = r2_value_sum/total_samples
    
    return metric, ci_value_sum, mae_value_sum, r2_value_sum

def train(data_args,train_args,model_args,writer,split_seeds=[2022,2023,2024,2025,2026], train_seed=400):
    epochs = train_args['epochs']                               
    device = train_args['device'] if torch.cuda.is_available() else 'cpu'
    save_path = train_args['save_path']

    wandb.config = train_args

    os.makedirs(save_path,exist_ok=True)
    
    all_results = {
        'dataset': train_args['data_name'],
        'split_seeds': split_seeds,
        'train_seed': train_seed,
        'folds': []
    }
    
   # preprocess = "n"  # By default, do not load preprocessed data for batch runs
    
    for seed in split_seeds:
        torch.manual_seed(train_seed)
        for fold in range(train_args['num_fold']):
            wandb.init(project='NedHGT', entity='entity_name',group=train_args["data_name"],name=f'seed{seed}_fold{fold}',reinit=True)
            if preprocess=='y':
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_train.pkl','rb') as f:
                    trainloader = dill.load(f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_valid.pkl','rb') as f:
                    valloader = dill.load(f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_test.pkl','rb') as f:
                    testloader = dill.load(f)
            else:
                trainloader = create_dataloader(data_args,f'{seed}_fold_{fold}_train.csv',shuffle=True)
                valloader = create_dataloader(data_args,f'{seed}_fold_{fold}_valid.csv',shuffle=False,train=False)
                testloader = create_dataloader(data_args,f'{seed}_fold_{fold}_test.csv',shuffle=False,train=False)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_train.pkl','wb') as f:
                    dill.dump(trainloader, f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_test.pkl','wb') as f:
                    dill.dump(testloader, f)
                with open(f'{train_args["data_name"]}_{seed}_fold_{fold}_valid.pkl','wb') as f:
                    dill.dump(valloader, f)
            print(f'dataset size, train: {len(trainloader.dataset)}, \
                    val: {len(valloader.dataset)}, \
                    test: {len(testloader.dataset)}')
            
            model = Model(model_args).to(device)
            optimizer = Adam(model.parameters())
            scheduler = NoamLR(
                optimizer=optimizer,
                warmup_epochs=[train_args['warmup']],
                total_epochs=[epochs],
                steps_per_epoch=len(trainloader.dataset) // data_args['batch_size'],
                init_lr=[train_args['init_lr']],
                max_lr=[train_args['max_lr']],
                final_lr=[train_args['final_lr']]
            )

            loss_fn = get_func(train_args['loss_fn'])
            metric_fn = get_func(train_args['metric_fn'])
            if train_args['loss_fn'] in []:
                loss_dtype = torch.long
            else:
                loss_dtype = torch.float32

            if train_args['metric_fn'] in []:
                metric_dtype = torch.long
            else:
                metric_dtype = torch.float32

            if train_args['metric_fn'] in ['auc','acc']:
                best = 0
                op = operator.ge
            else:
                best = np.inf
                op = operator.le
            best_epoch = 0
            
            patience = train_args.get('patience', 10)
            min_delta = train_args.get('min_delta', 0.0001)
            early_stop_counter = 0
            early_stop_enabled = train_args.get('early_stop', True)
            
            fold_results = {
                'seed': seed,
                'fold': fold,
                'train': {},
                'valid': {},
                'test': {}
            }
            
            for epoch in tqdm(range(epochs)):
                model.train()
                total_loss = 0
                for bg,labels,gf in trainloader:
                    bg,labels,gf = bg.to(device),labels.type(loss_dtype).to(device),gf.to(device)
                    pred = model(bg,gf)
                    num_task =  pred.size(1)
                    if num_task > 1:
                        loss = 0
                        for i in range(num_task):
                            loss += loss_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                    else:
                        loss = loss_fn(*remove_nan_label(pred,labels.reshape(pred.shape)))
                    total_loss += loss.item()*len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                total_loss = total_loss / len(trainloader.dataset)
                
                model.eval()
                with torch.no_grad():
                    val_rmse, val_ci, val_mae, val_r2 = evaluate(valloader,model,device,metric_fn,metric_dtype,data_args['task'])
                torch.cuda.empty_cache()
                
                improved = False
                if op(val_rmse, best):
                    if abs(val_rmse - best) >= min_delta or best == np.inf:
                        best = val_rmse
                        best_epoch = epoch
                        torch.save(model.state_dict(),os.path.join(save_path,f'./best_seed{seed}_fold{fold}.pt'))
                        early_stop_counter = 0
                        improved = True
                elif early_stop_enabled:
                    early_stop_counter += 1
                
                wandb.log({f'train {train_args["loss_fn"]} loss':round(total_loss,4),
                           f'valid RMSE': round(val_rmse,4),
                           'lr': round(math.log10(scheduler.lr[0]),4),
                           'CI': round(val_ci,4),
                           'MAE': round(val_mae,4),
                           'R2': round(val_r2,4),
                           'early_stop_counter': early_stop_counter,
                           })
                writer.add_scalar(tag="loss", 
                      scalar_value=total_loss,
                      global_step=epoch
                      )
                writer.add_scalar(tag="lr", 
                      scalar_value=math.log10(scheduler.lr[0]),
                      global_step=epoch
                      )
                writer.add_scalar(tag="Validate/RMSE", 
                      scalar_value=val_rmse,
                      global_step=epoch
                      )
                writer.add_scalar(tag="Validate/CI", 
                      scalar_value=val_ci,
                      global_step=epoch
                      )
                writer.add_scalar(tag="Validate/MAE", 
                      scalar_value=val_mae,
                      global_step=epoch
                      )
                writer.add_scalar(tag="Validate/R2", 
                      scalar_value=val_r2,
                      global_step=epoch
                      )
                writer.add_scalar(tag="EarlyStop/Counter", 
                      scalar_value=early_stop_counter,
                      global_step=epoch
                      )
                
                if early_stop_enabled and early_stop_counter >= patience:
                    print(f'\nEarly stopping triggered! Training stopped at epoch {epoch} (patience={patience})')
                    break
            
            model.eval()
            train_rmse, train_ci, train_mae, train_r2 = evaluate(trainloader,model,device,metric_fn,metric_dtype,data_args['task'])
            fold_results['train'] = {
                'RMSE': train_rmse,
                'CI': train_ci,
                'MAE': train_mae,
                'R2': train_r2
            }
            
            del model
            torch.cuda.empty_cache()
            
            model = Model(model_args).to(device)
            state_dict = torch.load(os.path.join(save_path,f'./best_seed{seed}_fold{fold}.pt'))
            model.load_state_dict(state_dict)
            model.eval()
            test_rmse, test_ci, test_mae, test_r2 = evaluate(testloader,model,device,metric_fn,metric_dtype,data_args['task'])
            
            model.eval()
            val_rmse_best, val_ci_best, val_mae_best, val_r2_best = evaluate(valloader,model,device,metric_fn,metric_dtype,data_args['task'])
            
            fold_results['valid'] = {
                'RMSE': val_rmse_best,
                'CI': val_ci_best,
                'MAE': val_mae_best,
                'R2': val_r2_best
            }
            fold_results['test'] = {
                'RMSE': test_rmse,
                'CI': test_ci,
                'MAE': test_mae,
                'R2': test_r2
            }
            fold_results['best_epoch'] = best_epoch
            
            all_results['folds'].append(fold_results)
            
            writer.add_scalar(tag="Test/RMSE", 
                      scalar_value=test_rmse,
                      global_step=fold
                      )
            writer.add_scalar(tag="Test/CI", 
                      scalar_value=test_ci,
                      global_step=fold
                      )
            writer.add_scalar(tag="Test/MAE", 
                      scalar_value=test_mae,
                      global_step=fold
                      )
            writer.add_scalar(tag="Test/R2", 
                      scalar_value=test_r2,
                      global_step=fold
                      )
            
            print(f'\n{"="*60}')
            print(f'Fold {fold} (seed={seed}, best_epoch={best_epoch})')
            print(f'  Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, CI={train_ci:.4f}, R2={train_r2:.4f}')
            print(f'  Valid: RMSE={val_rmse_best:.4f}, MAE={val_mae_best:.4f}, CI={val_ci_best:.4f}, R2={val_r2_best:.4f}')
            print(f'  Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, CI={test_ci:.4f}, R2={test_r2:.4f}')
            print(f'{"="*60}')
            
            wandb.finish()
    
    summary = calculate_summary(all_results)
    all_results['summary'] = summary
    
    results_file = os.path.join(save_path, f'{train_args["data_name"]}_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    csv_file = os.path.join(save_path, f'{train_args["data_name"]}_results.csv')
    save_results_csv(all_results, csv_file)
    
    print(f'\n{"="*60}')
    print(f'Summary Statistics (mean±std)')
    print(f'{"="*60}')
    print(f'  Train: RMSE={summary["train"]["RMSE"]["mean"]:.4f}±{summary["train"]["RMSE"]["std"]:.4f}, '
          f'MAE={summary["train"]["MAE"]["mean"]:.4f}±{summary["train"]["MAE"]["std"]:.4f}, '
          f'CI={summary["train"]["CI"]["mean"]:.4f}±{summary["train"]["CI"]["std"]:.4f}, '
          f'R2={summary["train"]["R2"]["mean"]:.4f}±{summary["train"]["R2"]["std"]:.4f}')
    print(f'  Valid: RMSE={summary["valid"]["RMSE"]["mean"]:.4f}±{summary["valid"]["RMSE"]["std"]:.4f}, '
          f'MAE={summary["valid"]["MAE"]["mean"]:.4f}±{summary["valid"]["MAE"]["std"]:.4f}, '
          f'CI={summary["valid"]["CI"]["mean"]:.4f}±{summary["valid"]["CI"]["std"]:.4f}, '
          f'R2={summary["valid"]["R2"]["mean"]:.4f}±{summary["valid"]["R2"]["std"]:.4f}')
    print(f'  Test:  RMSE={summary["test"]["RMSE"]["mean"]:.4f}±{summary["test"]["RMSE"]["std"]:.4f}, '
          f'MAE={summary["test"]["MAE"]["mean"]:.4f}±{summary["test"]["MAE"]["std"]:.4f}, '
          f'CI={summary["test"]["CI"]["mean"]:.4f}±{summary["test"]["CI"]["std"]:.4f}, '
          f'R2={summary["test"]["R2"]["mean"]:.4f}±{summary["test"]["R2"]["std"]:.4f}')
    print(f'{"="*60}')
    print(f'Results saved to:')
    print(f'  JSON: {results_file}')
    print(f'  CSV:  {csv_file}')
    
    return all_results

def calculate_summary(all_results):
    summary = {
        'train': {},
        'valid': {},
        'test': {}
    }
    
    metrics = ['RMSE', 'CI', 'MAE', 'R2']
    
    for phase in ['train', 'valid', 'test']:
        for metric in metrics:
            values = [fold[phase][metric] for fold in all_results['folds']]
            summary[phase][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    
    return summary

def save_results_csv(all_results, csv_file):
    rows = []
    for fold_data in all_results['folds']:
        row = {
            'seed': fold_data['seed'],
            'fold': fold_data['fold'],
            'best_epoch': fold_data['best_epoch'],
            'train_RMSE': fold_data['train']['RMSE'],
            'train_MAE': fold_data['train']['MAE'],
            'train_CI': fold_data['train']['CI'],
            'train_R2': fold_data['train']['R2'],
            'valid_RMSE': fold_data['valid']['RMSE'],
            'valid_MAE': fold_data['valid']['MAE'],
            'valid_CI': fold_data['valid']['CI'],
            'valid_R2': fold_data['valid']['R2'],
            'test_RMSE': fold_data['test']['RMSE'],
            'test_MAE': fold_data['test']['MAE'],
            'test_CI': fold_data['test']['CI'],
            'test_R2': fold_data['test']['R2'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    
    summary = all_results['summary']
    summary_row = {
        'seed': 'summary',
        'fold': 'mean±std',
        'best_epoch': '-',
        'train_RMSE': f'{summary["train"]["RMSE"]["mean"]:.4f}±{summary["train"]["RMSE"]["std"]:.4f}',
        'train_MAE': f'{summary["train"]["MAE"]["mean"]:.4f}±{summary["train"]["MAE"]["std"]:.4f}',
        'train_CI': f'{summary["train"]["CI"]["mean"]:.4f}±{summary["train"]["CI"]["std"]:.4f}',
        'train_R2': f'{summary["train"]["R2"]["mean"]:.4f}±{summary["train"]["R2"]["std"]:.4f}',
        'valid_RMSE': f'{summary["valid"]["RMSE"]["mean"]:.4f}±{summary["valid"]["RMSE"]["std"]:.4f}',
        'valid_MAE': f'{summary["valid"]["MAE"]["mean"]:.4f}±{summary["valid"]["MAE"]["std"]:.4f}',
        'valid_CI': f'{summary["valid"]["CI"]["mean"]:.4f}±{summary["valid"]["CI"]["std"]:.4f}',
        'valid_R2': f'{summary["valid"]["R2"]["mean"]:.4f}±{summary["valid"]["R2"]["std"]:.4f}',
        'test_RMSE': f'{summary["test"]["RMSE"]["mean"]:.4f}±{summary["test"]["RMSE"]["std"]:.4f}',
        'test_MAE': f'{summary["test"]["MAE"]["mean"]:.4f}±{summary["test"]["MAE"]["std"]:.4f}',
        'test_CI': f'{summary["test"]["CI"]["mean"]:.4f}±{summary["test"]["CI"]["std"]:.4f}',
        'test_R2': f'{summary["test"]["R2"]["mean"]:.4f}±{summary["test"]["R2"]["std"]:.4f}',
    }
    
    summary_df = pd.DataFrame([summary_row])
    df = pd.concat([df, summary_df], ignore_index=True)
    df.to_csv(csv_file, index=False)


if __name__=='__main__':

    import sys
    config_path = sys.argv[1]
    config = json.load(open(config_path,'r'))
    data_args = config['data']
    train_args = config['train']
    # If data_name is not in the config file, use the config file name
    if 'data_name' not in train_args:
        train_args['data_name'] = config_path.split('/')[-1].strip('.json').replace('\\', '/')
    model_args = config['model']
    
    split_seeds = config.get('seed', [2022, 2023, 2024, 2025, 2026])
    if not isinstance(split_seeds, list):
        split_seeds = [split_seeds]
    
    train_seed = 400
    # train_seed = 0
    # train_seed = 100
    # train_seed = 200
    # train_seed = 300
    
    print(config)
    comment = "data_{}&split_seeds_{}&train_seed_{}".format(train_args['data_name'], split_seeds, train_seed)
    writer = SummaryWriter(comment=comment)
    results = train(data_args, train_args, model_args, writer, split_seeds, train_seed)
