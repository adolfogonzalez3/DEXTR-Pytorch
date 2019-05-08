import os.path

from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
import dataloaders.pascal as pascal
from eval_model import eval_model

from nets import create_densenet, create_squeezenet, create_shufflenet

exp_root_dir = Path('RUNS', 'PARAMSEARCH')

method_names = []
#method_names.append('run_-1')

def load_model(model_name, nInputChannels):
    if model_name == 'densenet':
        return create_densenet(nInputChannels)
    elif model_name == 'squeezenet':
        return create_squeezenet(nInputChannels)
    elif model_name == 'shufflenet':
        return create_shufflenet(nInputChannels)


if __name__ == '__main__':

    # Dataloader
    dataset = pascal.VOCSegmentation(transform=None, retname=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Iterate through all the different methods
    #for method in method_names:
    eval_folder = exp_root_dir / 'eval_results'
    eval_folder.mkdir(exist_ok=True)
    exp_folders = [d for d in exp_root_dir.iterdir()
                   if d.name.startswith('run')]
    exp_folders.reverse()
    for project_folder in tqdm(exp_folders, desc="Testing..."):
        method = project_folder.name.split('-')[1]
        #results_folder = os.path.join(exp_root_dir, method, 'Results')
        results_folder = project_folder / 'Results'
        models_folder = project_folder / 'models'

        #filename = os.path.join(exp_root_dir, 'eval_results', method.replace('/', '-') + '.txt')
        filename = eval_folder / (project_folder.name + '.txt')
        #if not os.path.exists(os.path.join(exp_root_dir, 'eval_results')):
        #    os.makedirs(os.path.join(exp_root_dir, 'eval_results'))

        tqdm.write(str(filename))

        #if os.path.isfile(filename):
        if filename.exists():
            #with open(filename, 'r') as f:
            with filename.open('r') as f:
                val = float(f.read())
        else:
            if not results_folder.exists():
                latest_weight = max([int(w.stem.split('-')[-1])
                                     for w in models_folder.iterdir()
                                     if w.name.endswith('.pth')])
                weight_files = [str(w) for w in models_folder.iterdir()
                                if w.name.endswith('.pth')]
                net = load_model(method, 4)
                device = torch.device('cuda:2')
                net.to(device)
                wght_name = '{}_epoch-{:d}.pth'.format(method, latest_weight)
                net.load_state_dict(torch.load(str(models_folder / wght_name),
                                               map_location=device))
                eval_model(net, results_folder, batch_size=10)
                    
            tqdm.write("Evaluating method: {}".format(method))
            jaccards = eval_one_result(dataloader, str(results_folder),
                                       mask_thres=0.8)
            val = jaccards["all_jaccards"].mean()

        # Show mean and store result
        tqdm.write("Result for {:<80}: {}".format(method, str.format("{0:.1f}", 100*val)))
        with open(filename, 'w') as f:
            f.write(str(val))
