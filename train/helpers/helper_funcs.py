import torch
import numpy as np
import h5py
import os
from sklearn.linear_model import LinearRegression

CONDS_TEMPLATE = '/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{n}.npy'
# exclude_{split}.npy index files written by helpers/scene_overlap.py --exclusion_dir
HELDOUT_DIR = '/share/klab/psulewski/psulewski/memdur_paper/data/gpn_heldout'

##############################
## Split parsing / held-out images
##############################

def parse_split(split):
    """
    'train_515_noavs' -> ('train', 515, True);  'val_noavs' -> ('val', None, True);
    'test_515'        -> ('test', 515, False);  'train'     -> ('train', None, False).
    """
    parts = split.split('_')
    exclude_avs = parts[-1] == 'noavs'
    if exclude_avs:
        parts = parts[:-1]
    base = parts[0]
    if base not in ('train', 'val', 'test'):
        raise ValueError(f'unknown split {split!r}')
    test_case = int(parts[1]) if len(parts) > 1 else None
    return base, test_case, exclude_avs


def load_avs_exclusions(heldout_dir=HELDOUT_DIR):
    """
    Per-split h5 row indices of AVS-MEG stimulus scenes, to be held out of training.
    Written by `python -m helpers.scene_overlap --exclusion_dir <dir>`.
    """
    exclusions = {}
    for src in ('train', 'val', 'test'):
        path = os.path.join(heldout_dir, f'exclude_{src}.npy')
        if os.path.exists(path):
            exclusions[src] = np.load(path)
    if not exclusions:
        raise FileNotFoundError(
            f'no exclude_{{split}}.npy found in {heldout_dir}. Generate them with:\n'
            f'  python -m helpers.scene_overlap --exclusion_dir {heldout_dir}')
    return exclusions


def _drop(idx, exclude):
    """Remove `exclude` from the sorted index array `idx`."""
    if exclude is None or not len(exclude):
        return np.asarray(idx)
    return np.setdiff1d(np.asarray(idx), np.asarray(exclude))


##############################
## Loading the dataset loaders
##############################

def get_Dataset_loaders(hyp,split):

    dataset_path = hyp['dataset']['dataset_path']
    gaze_type = hyp['network']['gaze_type']
    timesteps = hyp['network']['timesteps']

    split_data = CocoGaze(split=split, dataset_path=dataset_path, gaze_type=gaze_type, timesteps=timesteps, in_memory= hyp['dataset']['in_memory'], bbv=hyp['dataset']['bbv'], dva_dataset=hyp['dataset']['dva_dataset'], heldout_dir=hyp['dataset'].get('heldout_dir', HELDOUT_DIR))

    if 'train' in split:
        data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=True,
                                                    num_workers=hyp['optimizer']['dataloader']['num_workers_train'],
                                                    prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
    elif 'val' in split or 'test' in split:
        data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=False,
                                                    num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                    prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
        
    return data_loader

class CocoGaze(torch.utils.data.Dataset):
    """
    Glimpse-sequence dataset, addressed as (image, gaze trace) pairs.

    Split names
    -----------
    'train' / 'val' / 'test'   one h5 split as-is
    'train_{N}'                the train split + the test split minus pythonic_conds{N}
    'test_{N}'                 only pythonic_conds{N} of the test split
    '..._noavs'                additionally drops every image whose COCO id is in the AVS-MEG
                               stimulus set, using exclude_{split}.npy index files written by
                               helpers/scene_overlap.py. 87.4% of the AVS scenes were otherwise
                               training items (all of them sit in the test split, of which
                               train_515 holds out only 515), so this is the leakage control.

    Rows are resolved once in __init__ into a flat (source split, source row) index, so the
    in-memory and on-disk paths share one addressing scheme and nothing is recomputed per item.
    """

    def __init__(self, split, dataset_path, gaze_type, timesteps, in_memory, bbv,
                 dva_dataset='NSD', heldout_dir=HELDOUT_DIR):

        self.root_dir = dataset_path
        gaze_map = {'dg3': 0, 'random': 1, 'dg3p': 2, 'dg3r': 3}
        self.gaze_type = gaze_map[gaze_type] # 0 if dg3, 1 if random, 2 if dg3_permuted, 3 if dg3_swap
        self.in_memory = in_memory
        self.split = split
        self.timesteps = timesteps
        self.dva_dataset = dva_dataset
        if self.dva_dataset == 'NSD':
            self.dataset_str = f'coco_NSD_dg3fix91_r50v{bbv}ap_7fix'
        elif self.dva_dataset == 'AVS':
            self.dataset_str = f'coco_AVS_dg3fix36_r50v{bbv}ap_7fix'
        self.coco_dataset_str = dataset_path + '../optimized_datasets/ms_coco_embeddings.h5'

        base, test_case, exclude_avs = parse_split(split)
        exclusions = load_avs_exclusions(heldout_dir) if exclude_avs else {}

        # how many images each h5 split holds (cheap metadata read)
        n_imgs = {}
        for src in set([base] + (['test'] if test_case is not None else [])):
            with h5py.File(dataset_path + f'{self.dataset_str}_{src}.h5', 'r') as f:
                n_imgs[src] = f[src]['mpnet_embeddings'].shape[0]

        # (source split, row indices) segments making up this dataset
        if test_case is None:
            segments = [(base, _drop(np.arange(n_imgs[base]), exclusions.get(base)))]
        elif base == 'train':
            # the train_{N} trainer: all of train + all of test except the N special images
            conds = np.load(CONDS_TEMPLATE.format(n=test_case))
            segments = [('train', _drop(np.arange(n_imgs['train']), exclusions.get('train'))),
                        ('test', _drop(np.setdiff1d(np.arange(n_imgs['test']), conds),
                                       exclusions.get('test')))]
        else:
            conds = np.load(CONDS_TEMPLATE.format(n=test_case))
            segments = [('test', _drop(conds, exclusions.get('test')))]

        self.segments = [(src, idx) for src, idx in segments if len(idx)]
        if not self.segments:
            raise ValueError(f'split {split!r} selects no images')
        self.src_split = np.concatenate([np.full(len(idx), i, dtype=np.int16)
                                         for i, (_, idx) in enumerate(self.segments)])
        self.src_row = np.concatenate([idx for _, idx in self.segments])

        with h5py.File(dataset_path + f'{self.dataset_str}_{self.segments[0][0]}.h5', 'r') as f:
            n_traces = f[self.segments[0][0]]['dg3_fix_actvs'].shape[1]
        self.len = (len(self.src_row), n_traces)

        print(f'CocoGaze {split}: {self.len[0]} images x {self.len[1]} traces from '
              + ', '.join(f'{src} ({len(idx)})' for src, idx in self.segments)
              + (f' | AVS-excluded images: '
                 + ', '.join(f'{k} {len(v)}' for k, v in sorted(exclusions.items()))
                 if exclusions else ''))

        if in_memory == 1:
            self._load_in_memory()

    def _read(self, f, src, name, idx, trailing=()):
        """Read `name` for rows `idx`, using a plain slice when the whole split is kept."""
        dset = f[src][name]
        whole = len(idx) == dset.shape[0] and idx[0] == 0 and idx[-1] == dset.shape[0] - 1
        sel = (slice(None),) if whole else (idx,)
        return dset[sel + trailing]

    # (attribute, h5 dataset, trailing slicer) — the trailing slicer picks this run's
    # timesteps and gaze type, exactly as the original per-branch reads did
    def _fields(self):
        T, g = self.timesteps, self.gaze_type
        return [
            ('actvs', 'dg3_fix_actvs',
             (slice(None), slice(0, T + 1), g, slice(None))),
            ('next_fix_rel_coords', 'next_fix_rel_coords',
             (slice(None), slice(0, T), g, slice(None))),
            ('next_fix_coords', 'next_fix_coords',
             (slice(None), slice(0, T), g, slice(None))),
            ('semantic_embed', 'mpnet_embeddings', ()),
            ('scene_embed', 'full_image_actvs', ()),
        ]

    def _load_in_memory(self):
        """
        Fill one preallocated float32 tensor per field, segment by segment. Preallocating
        keeps the peak at one target tensor plus one segment's float16 read buffer — the
        same profile as the original per-branch code (the full train_515 actvs tensor is
        ~70 GB in float32, so request memory accordingly, or use in_memory=0).
        """
        n_total = self.len[0]
        offset = 0
        for src, idx in self.segments:
            with h5py.File(self.root_dir + f'{self.dataset_str}_{src}.h5', 'r') as f:
                for attr, name, trailing in self._fields():
                    arr = self._read(f, src, name, idx, trailing)
                    if not hasattr(self, attr):
                        setattr(self, attr, torch.empty((n_total,) + arr.shape[1:],
                                                        dtype=torch.float32))
                    getattr(self, attr)[offset:offset + len(idx)] = torch.from_numpy(arr)
                    del arr
            with h5py.File(self.coco_dataset_str, 'r') as f_coco:
                # float32 because the scene loss consumes it as a cosine-similarity target
                arr = self._read(f_coco, src, 'img_multi_hot', idx)
                if not hasattr(self, 'scene_multihot'):
                    self.scene_multihot = torch.empty((n_total,) + arr.shape[1:],
                                                      dtype=torch.float32)
                self.scene_multihot[offset:offset + len(idx)] = torch.from_numpy(arr)
                del arr
            offset += len(idx)

    def __len__(self):
        return (self.len[0]*self.len[1]) # each image * gaze trace is considered a sample

    def __getitem__(self, idx): # accepts ids for img/trace and returns for fixations, the actvs, fixs_coords, to the Dataloader

        img_n = idx//self.len[1]
        trace_n = idx%self.len[1]

        if self.in_memory == 1:

            actvs = self.actvs[img_n,trace_n,:,:]
            next_fix_rel_coords = self.next_fix_rel_coords[img_n,trace_n,:,:]
            fix_coords = self.next_fix_coords[img_n,trace_n,:,:]
            fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
            semantic_embed = self.semantic_embed[img_n,:]
            scene_embed = self.scene_embed[img_n,:]
            scene_multihot = self.scene_multihot[img_n,:]

        else:

            src = self.segments[int(self.src_split[img_n])][0]
            row = int(self.src_row[img_n])
            T = self.timesteps
            g = self.gaze_type

            with h5py.File(self.root_dir + f'{self.dataset_str}_{src}.h5', "r") as f:
                actvs = torch.from_numpy(f[src]['dg3_fix_actvs'][row,trace_n,:T+1,g,:][()]).float()
                next_fix_rel_coords = torch.from_numpy(f[src]['next_fix_rel_coords'][row,trace_n,:T,g,:][()]).float()
                fix_coords = torch.from_numpy(f[src]['next_fix_coords'][row,trace_n,:T,g,:][()]).float()
                fix_coords = torch.cat((torch.Tensor([[0, 0]]), fix_coords), dim=0) # adding center coordinates to tensor
                semantic_embed = torch.from_numpy(f[src]['mpnet_embeddings'][row,:][()]).float()
                scene_embed = torch.from_numpy(f[src]['full_image_actvs'][row,:][()]).float()
            with h5py.File(self.coco_dataset_str, 'r') as f_coco:
                scene_multihot = torch.from_numpy(f_coco[src]['img_multi_hot'][row,:][()]).float()

        img_n = torch.Tensor([img_n])
        trace_n = torch.Tensor([trace_n])

        return actvs, next_fix_rel_coords, fix_coords, semantic_embed, scene_embed, scene_multihot, img_n, trace_n
    
def get_Dataset_loaders_e2e(hyp,split,preprocess,img_scale=224):
    gaze_type = hyp['network']['gaze_type']
    timesteps = hyp['network']['timesteps']

    split_data = CocoGaze_e2e(split=split, gaze_type=gaze_type, timesteps=timesteps, in_memory=hyp['dataset']['in_memory'],dva_dataset=hyp['dataset']['dva_dataset'],preprocess=preprocess,img_scale=img_scale)

    if 'train' in split:
        data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=True, num_workers=hyp['optimizer']['dataloader']['num_workers_train'], prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
    elif 'val' in split or 'test' in split:
        data_loader = torch.utils.data.DataLoader(split_data, batch_size=hyp['optimizer']['batch_size'], shuffle=False, num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'], prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor'])
        
    return data_loader
    
class CocoGaze_e2e(torch.utils.data.Dataset):
    #Import dataset splitwise

    def __init__(self, split, gaze_type, timesteps, in_memory, dva_dataset='NSD', preprocess=None,img_scale=224): # only in_memory, dg3 agze_type, implemented here!

        self.gaze_type = 'dg3'
        self.in_memory = in_memory
        self.split = split
        self.timesteps = timesteps
        self.dva_dataset = dva_dataset
        if self.dva_dataset == 'NSD':
            self.glimpse_size = 91
        elif self.dva_dataset == 'AVS':
            self.glimpse_size = 36
        self.preprocess = preprocess
        self.img_scale = img_scale

        f_dg3_fixations = h5py.File('/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5', 'r')
        f_coco = h5py.File('/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5', 'r')

        if split == 'train' or split == 'val' or split == 'test':

            self.coco_imgs = torch.from_numpy(f_coco[split]['data'][()])
            self.dg3_fixs = torch.from_numpy(f_dg3_fixations[split]['densenet_deepgaze_fixations'][()])
        
        elif len(split.split('_')) == 2 and split.split('_')[0] == 'train':

            test_case = int(split.split('_')[1])

            n_imgs_tt = f_coco['test']['data'].shape[0]
            idxs_special = np.load(f'/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{test_case}.npy')

            if split.split('_')[0] == 'train':

                self.coco_imgs = torch.from_numpy(np.concatenate((f_coco['train']['data'][()], np.array([f_coco['test']['data'][i] for i in range(n_imgs_tt) if i not in list(idxs_special)])), axis=0))
                self.dg3_fixs = torch.from_numpy(np.concatenate((f_dg3_fixations['train']['densenet_deepgaze_fixations'][()], np.array([f_dg3_fixations['test']['densenet_deepgaze_fixations'][i] for i in range(n_imgs_tt) if i not in list(idxs_special)])), axis=0))

            elif split.split('_')[0] == 'test':

                self.coco_imgs = torch.from_numpy(np.array([f_coco['test']['data'][i] for i in list(idxs_special)]))
                self.dg3_fixs = torch.from_numpy(np.array([f_dg3_fixations['test']['densenet_deepgaze_fixations'][i] for i in list(idxs_special)]))

    def __len__(self):
        return (self.dg3_fixs.shape[0]*self.dg3_fixs.shape[1]) # each image * gaze trace is considered a sample
    
    def __getitem__(self, idx): # accepts ids for img/trace and returns for glimpse sequences to the Dataloader

        trace_n = idx//self.dg3_fixs.shape[0]
        img_n = idx%self.dg3_fixs.shape[0]

        glimpse_ext = self.glimpse_size

        img = self.coco_imgs[img_n,:,:,:]
        im_h = torch.zeros((int(img.shape[0]+glimpse_ext),int(img.shape[1]+glimpse_ext),3),dtype=img.dtype)
        im_h[int(glimpse_ext/2):int(glimpse_ext/2)+img.shape[0],int(glimpse_ext/2):int(glimpse_ext/2)+img.shape[1],:] = img
        img = im_h
        
        dg3_fixs = self.dg3_fixs[img_n,trace_n,:,:]
    
        glimpse_seq = torch.zeros((self.timesteps+1, 3, self.img_scale, self.img_scale))
        next_fix_rel_coords = torch.zeros((self.timesteps, 2))

        for gli_idx in range(self.timesteps+1):
            
            x_ext_low = int(dg3_fixs[gli_idx,0])
            x_ext_high = int(dg3_fixs[gli_idx,0]+glimpse_ext)
            y_ext_low = int(dg3_fixs[gli_idx,1])
            y_ext_high = int(dg3_fixs[gli_idx,1]+glimpse_ext)

            glimpse = img[y_ext_low:y_ext_high,x_ext_low:x_ext_high,:]
            glimpse = glimpse.permute(2,0,1)
            if self.preprocess is not None:
                glimpse = self.preprocess(glimpse)
            glimpse_seq[gli_idx,:,:,:] = glimpse

            if gli_idx < self.timesteps:
                next_fix_rel_coords[gli_idx,:] = dg3_fixs[gli_idx+1,:] - dg3_fixs[gli_idx,:]

        img_n = torch.Tensor([img_n])
        trace_n = torch.Tensor([trace_n])

        return glimpse_seq, next_fix_rel_coords, img_n, trace_n
    
def create_cpc_matrix(N, m, mask=True):
    # mask==True creates a mask matrix; if False creates a matrix that can be used directly for loss

    # Calculate the block value and the rest value
    if mask:
        block_value = 2
        rest_value = 3
        diag_value = 1
    else:
        block_value = 0.5 * (1 / ((N/m) * (m**2 - m)))
        rest_value = 0.5 * (1 / (N**2 - (N/m) * (m**2)))
        diag_value = -1/N
    
    # Initialize the matrix with the rest_value
    matrix = torch.full((N, N), rest_value, requires_grad=False)
    
    # Set the diagonal elements
    torch.diagonal(matrix).fill_(diag_value)
    
    # Fill the blocks along the diagonal
    for i in range(0, N, m):
        matrix[i:i+m, i:i+m] = torch.full((m, m), block_value)
        torch.diagonal(matrix[i:i+m, i:i+m]).fill_(diag_value)

    return matrix

class LinearFitScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, factor=1./2, min_lr=1e-8, min_percent_change=1.0, mode='min', patience=5, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_epochs (int): Number of epochs to use for the linear fit.
            factor (float): Factor by which the learning rate will be reduced. Default: 0.1.
            min_lr (float): Minimum learning rate. Default: 1e-6.
            min_percent_change (float): Minimum absolute percentage change in the metric to not trigger a reduction. Default: 1.0.
            mode (str): One of 'min' or 'max'. 'min' will reduce the LR if the metric has not decreased by min_percent_change,
                        'max' will reduce the LR if the metric has not increased by min_percent_change. Default: 'min'.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default: 0.
            last_epoch (int): The index of the last epoch. Default: -1.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        self.num_epochs = num_epochs
        self.factor = factor
        self.min_lr = min_lr
        self.min_percent_change = min_percent_change
        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0  # Track the number of epochs without improvement
        self.verbose = verbose
        self.metric_history = []
        super(LinearFitScheduler, self).__init__(optimizer, last_epoch)

    def step(self, metric=None):
        """
        Step should be called after each epoch. Can be called without 'metric' during initialization.
        
        Args:
            metric (float, optional): Current epoch's metric. Default is None.
        """
        # Increment the last_epoch attribute from the base class
        self.last_epoch += 1
        
        if metric is not None:
            # Update metric history
            self.metric_history.append(metric)
            
            # Only perform the check if we have enough history
            if len(self.metric_history) >= self.num_epochs:
                # Perform linear fit
                epochs = np.arange(self.num_epochs).reshape(-1, 1)
                metrics = np.array(self.metric_history[-self.num_epochs:]).reshape(-1, 1)
                
                reg = LinearRegression().fit(epochs, metrics)
                slope = reg.coef_[0, 0]
                intercept = reg.intercept_[0]
                
                # Calculate the predicted metrics
                predicted_start = intercept
                predicted_end = slope * (self.num_epochs - 1) + intercept
                
                # Calculate percent change based on the magnitude of the start value
                if predicted_start != 0:
                    percent_change = 100 * (predicted_end - predicted_start) / abs(predicted_start)
                else:
                    percent_change = float('inf')  # Avoid division by zero
                if self.verbose:
                    print(f"Percent_change in metric: {percent_change:.2f}%")
                    
                # Determine if we should adjust the learning rate based on the mode and percent change
                if self.mode == 'min' and percent_change > -self.min_percent_change:
                    self.num_bad_epochs += 1
                elif self.mode == 'max' and percent_change < self.min_percent_change:
                    self.num_bad_epochs += 1
                else:
                    self.num_bad_epochs = 0  # Reset counter if improvement is observed
                
                # Check if we have hit the patience threshold
                if self.num_bad_epochs > self.patience:
                    self.reduce_lr(percent_change)
                    self.num_bad_epochs = 0  # Reset bad epoch count after reducing LR

    def reduce_lr(self, percent_change):
        """Reduce the learning rate according to the factor and min_lr constraints and print verbose message."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing learning rate of group {i} to {new_lr:.4e}. Percent change: {percent_change:.2f}%. Patience exceeded.")
    
    
##############################
## Logging functions
##############################
    
def create_folders_logging(net_name, create=1):

    print('Accessing log folders...')

    log_folder = 'logs/perf_logs'
    net_folder = 'logs/net_params'

    if create:
        isExist = os.path.exists(log_folder)
        if not isExist:
            os.makedirs(log_folder)
            print('Log folder is created!')
        isExist = os.path.exists(net_folder)
        if not isExist:
            os.makedirs(net_folder)
            print('Net folder is created!')

    log_folder_name = log_folder+f'/{net_name}'
    net_folder_name = net_folder+f'/{net_name}'

    if create:
        isExist = os.path.exists(log_folder_name)
        if not isExist:
            os.makedirs(log_folder_name)
            print('Specific log folder is created!')
        isExist = os.path.exists(net_folder_name)
        if not isExist:
            os.makedirs(net_folder_name)
            print('Specific net folder is created!')

    return log_folder_name, net_folder_name 