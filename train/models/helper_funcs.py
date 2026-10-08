import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
    
##################################
## Importing the network
##################################

def get_network_model(hyp):
    # import the req. network

    timestep_multiplier = hyp['network']['timestep_multiplier']
    timesteps = hyp['network']['timesteps']
    gaze_type = hyp['network']['gaze_type']
    network_id = hyp['network']['identifier']
    n_rnn = hyp['network']['n_rnn']
    regularisation = hyp['network']['regularisation']
    input_dropout = hyp['network']['input_dropout']
    rnn_dropout = hyp['network']['rnn_dropout']
    analysis_mode = hyp['network']['analysis_mode']
    input_split = hyp['network']['input_split']
    recurrence = hyp['network']['recurrence']

    semantic_loss = hyp['optimizer']['losses']['semantic_loss']
    scene_loss = hyp['optimizer']['losses']['scene_loss']
    glimpse_loss = hyp['optimizer']['losses']['glimpse_loss']
    gazeloc_loss = hyp['optimizer']['losses']['gazeloc_loss']
    provide_loc = hyp['optimizer']['losses']['provide_loc']

    bbv = hyp['dataset']['bbv']
    print(f'\nUsing bbv-{bbv} features')
    dva_dataset = hyp['dataset']['dva_dataset']

    trainer = hyp['optimizer']['trainer']
    lr = hyp['optimizer']['lr']

    # the InfoNCE semantic loss carries its temperature in the name, so a temperature sweep
    # separates into its own checkpoints/logs/W&B runs; empty for the published variants, which
    # keeps their names (and `_semc_(\d+)_` parsing downstream) byte-identical
    sem_tag = ''
    if semantic_loss == 2:
        sem_tag = f"_semtau_{hyp['optimizer']['losses']['semantic_temp']}_semnorm_{hyp['optimizer']['losses']['semantic_norm']}_semdd_{hyp['optimizer']['losses']['semantic_dedup']}"

    if hyp['network']['model'] == 'lstm':

        from .GPN import lstm_gpn

        net = lstm_gpn(timestep_multiplier=timestep_multiplier,glimpse_loss=glimpse_loss,semantic_loss=semantic_loss,scene_loss=scene_loss,gazeloc_loss=gazeloc_loss,n_rnn=n_rnn,regularisation=regularisation,input_dropout=input_dropout,rnn_dropout=rnn_dropout,return_all_actvs=analysis_mode, input_split=input_split, recurrence=recurrence, input_feats=768 if bbv == 5 else 2048)

        net_name = f'gpn_lstm_n_{n_rnn}_tm_{timestep_multiplier}_t_{timesteps}_recurrence_{recurrence}_loc_{provide_loc}_bbv{bbv}_gaze_{gaze_type}_indp_{input_dropout}_rnndp_{rnn_dropout}_gcpc_{glimpse_loss}_semc_{semantic_loss}{sem_tag}_scc_{scene_loss}_locmse_{gazeloc_loss}_insplit_{input_split}_reg_{regularisation}_tr_{trainer}_gdva_{dva_dataset}_lr_{lr}_num_{network_id}'

    elif hyp['network']['model'] == 'gru':

        from .GPN import gru_gpn

        net = gru_gpn(timestep_multiplier=timestep_multiplier,glimpse_loss=glimpse_loss,semantic_loss=semantic_loss,scene_loss=scene_loss,gazeloc_loss=gazeloc_loss,n_rnn=n_rnn,regularisation=regularisation,input_dropout=input_dropout,rnn_dropout=rnn_dropout,return_all_actvs=analysis_mode, input_split=input_split, recurrence=recurrence, input_feats=768 if bbv == 5 else 2048)

        net_name = f'gpn_gru_n_{n_rnn}_tm_{timestep_multiplier}_t_{timesteps}_recurrence_{recurrence}_loc_{provide_loc}_bbv{bbv}_gaze_{gaze_type}_indp_{input_dropout}_rnndp_{rnn_dropout}_gcpc_{glimpse_loss}_semc_{semantic_loss}{sem_tag}_scc_{scene_loss}_locmse_{gazeloc_loss}_insplit_{input_split}_reg_{regularisation}_tr_{trainer}_gdva_{dva_dataset}_lr_{lr}_num_{network_id}'

    print(f'\nNetwork name: {net_name}')

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"\nThe network has {params} trainable parameters\n")

    return net, net_name

def get_network_model_e2e(hyp):
    # import the req. network

    timestep_multiplier = hyp['network']['timestep_multiplier']
    timesteps = hyp['network']['timesteps']
    gaze_type = hyp['network']['gaze_type']
    network_id = hyp['network']['identifier']
    n_rnn = hyp['network']['n_rnn']
    regularisation = hyp['network']['regularisation']
    input_dropout = hyp['network']['input_dropout']
    rnn_dropout = hyp['network']['rnn_dropout']
    analysis_mode = hyp['network']['analysis_mode']
    input_split = hyp['network']['input_split']
    recurrence = hyp['network']['recurrence']

    semantic_loss = hyp['optimizer']['losses']['semantic_loss']
    scene_loss = hyp['optimizer']['losses']['scene_loss']
    glimpse_loss = hyp['optimizer']['losses']['glimpse_loss']
    gazeloc_loss = hyp['optimizer']['losses']['gazeloc_loss']
    provide_loc = hyp['optimizer']['losses']['provide_loc']

    dva_dataset = hyp['dataset']['dva_dataset']

    trainer = hyp['optimizer']['trainer']
    lr = hyp['optimizer']['lr']

    if hyp['network']['model'] == 'rn18-lstm':

        from .GPN import rn18_lstm_gpn

        net = rn18_lstm_gpn(timestep_multiplier=timestep_multiplier,glimpse_loss=glimpse_loss,semantic_loss=semantic_loss,scene_loss=scene_loss,gazeloc_loss=gazeloc_loss,n_rnn=n_rnn,regularisation=regularisation,input_dropout=input_dropout,rnn_dropout=rnn_dropout,return_all_actvs=analysis_mode, input_split=input_split, recurrence=recurrence) 

        net_name = f'gpn_e2e_rn18_lstm_n_{n_rnn}_tm_{timestep_multiplier}_t_{timesteps}_recurrence_{recurrence}_loc_{provide_loc}_gaze_{gaze_type}_indp_{input_dropout}_rnndp_{rnn_dropout}_reg_{regularisation}_tr_{trainer}_gdva_{dva_dataset}_lr_{lr}_num_{network_id}'

    print(f'\nNetwork name: {net_name}')

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"\nThe network has {params} trainable parameters\n")

    return net, net_name

def weights_init(m):
    # Xavier intialisation for conv and linear - LSTM has self-initialisation
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def get_optimizer(hyp,net):
    # selecting the optimizer

    if hyp['optimizer']['type'] == 'adam': # write an optimizer with access to the entire net and another for finetuning desired outputs
        return optim.Adam(net.parameters(),lr=1.)
    
def semantic_infonce(similarity_matrix, cpc_mask, temperature, norm='tau', same_img=None):
    """
    Temperature-scaled InfoNCE over the semantic (caption-embedding) similarity matrix.

    The published semantic loss is a flat mean-difference,

        sim[cpc_mask==3].mean() - sim[cpc_mask==1].mean()

    which weights every cross-scene negative equally. This replaces it with a symmetric
    softmax-with-temperature over exactly the same entries, so a negative contributes in
    proportion to how hard it is: visually/semantically near-confusable scenes dominate the
    gradient, near-orthogonal ones drop out.

    Candidate set per row = the positive (the diagonal) plus the cross-scene negatives
    (cpc_mask==3). Within-sequence off-diagonal entries (cpc_mask==2) are *excluded*, not
    used as negatives, because A repeats one caption embedding across the timesteps of a
    sequence, so those entries compare a target with itself. That matches the published
    loss, which also only ever touches masks 1 and 3.

    Symmetric = mean of the two directions (target->prediction and prediction->target).
    The published loss averages over both triangles of the matrix, so a symmetric InfoNCE
    keeps the change confined to "softmax instead of flat mean".

    norm controls the constant the raw cross-entropy (in nats) is rescaled by. It cannot
    change the gradient direction, only its scale, but it decides whether a temperature
    sweep is interpretable and whether the percent-based LR scheduler / early stopping in
    train_net.py keep behaving as they do on the published loss:

      'tau'  (default)  tau * (nce - log K). Chance = 0, solved ~= -tau*log K, i.e. the same
                        sign and roughly the same dynamic range as the published loss. The
                        1/tau in dInfoNCE/dsim cancels, so gradient magnitude is
                        tau-independent (no effective-LR confound across a tau sweep), and
                        as tau -> inf the gradient converges to that of the flat
                        mean-difference loss -- the published objective is the tau -> inf
                        limit of this one.
      'logk'            nce/log K - 1. Chance = 0, solved = -1, comparable across tau in
                        *loss* units rather than gradient units.
      'none'            raw nats, the standard InfoNCE number. Note this makes the loss
                        positive, which inverts the meaning of the `val > 0.99*min(val)`
                        early-stopping rule in train_net.py -- only use it for diagnostics.

    same_img: optional (N, N) bool, True where two rows come from the same scene. A batch
    holds several scanpaths per scene, so two different rows can carry an *identical*
    caption embedding (~1 colliding pair per batch of 512 for train_515). The flat mean
    barely notices; a low-temperature softmax hands such an exact duplicate half the mass
    and then pushes the prediction away from an identical target, so those entries are
    dropped from the denominator.

    Returns (loss, diagnostics) with diagnostics detached, in nats / fractions.
    """
    logits = similarity_matrix.float() / temperature
    valid = (cpc_mask == 1) | (cpc_mask == 3)
    if same_img is not None:
        valid = valid & (~same_img | (cpc_mask == 1))  # never drop the positive itself
    logits = logits.masked_fill(~valid, float('-inf'))

    pos = logits.diagonal()
    n_valid = valid.sum(dim=1).float()  # cpc_mask and same_img are symmetric, so rows == cols
    nce = 0.5 * ((logits.logsumexp(dim=1) - pos) + (logits.logsumexp(dim=0) - pos))
    chance = torch.log(n_valid)

    if norm == 'tau':
        loss = (temperature * (nce - chance)).mean()
    elif norm == 'logk':
        loss = (nce / chance - 1.).mean()
    elif norm == 'none':
        loss = nce.mean()
    else:
        raise ValueError(f'unknown semantic_norm: {norm}')

    with torch.no_grad():
        n = logits.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=logits.device)
        top1 = (logits.argmax(dim=1) == torch.arange(n, device=logits.device)).float().mean()
        # how concentrated the negative gradient is: 1 = uniform over negatives (what the
        # published flat mean does), -> 0 = all of it on a handful of hard negatives
        neg_logp = logits.masked_fill(eye, float('-inf'))
        neg_logp = neg_logp - neg_logp.logsumexp(dim=1, keepdim=True)
        ent = -torch.nan_to_num(neg_logp.exp() * neg_logp, nan=0.).sum(dim=1)
        diagnostics = {
            'sem/nats': nce.mean().item(),
            'sem/nats_chance': chance.mean().item(),
            'sem/top1': top1.item(),
            'sem/neg_entropy_frac': (ent / torch.log(n_valid - 1.)).mean().item(),
            'sem/pos_sim': similarity_matrix.float().diagonal().mean().item(),
        }
    return loss, diagnostics


def compute_losses(outputs,actvs,fix_coords,semantic_embed,scene_embed,cpc_mask,hyp,compute_contrastive_floor,img_n=None):

    loss_combined = 0.
    contrastive_loss_floor = 0.
    diagnostics = {}

    if hyp['optimizer']['losses']['glimpse_loss'] == 1:

        A = actvs[:,1:,:].reshape(-1, outputs[0].shape[2])
        B = outputs[0].reshape(-1, outputs[0].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T
        A_filter_map = ((A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T) < 0.999 # places where activations are not repeated or EXTREMELY similar - else they make the loss inelegant
        # MSE: similarity_matrix = -torch.sqrt(torch.sum(A**2, dim=1, keepdim=True) + torch.sum(B**2, dim=1).unsqueeze(0) - 2 * torch.mm(A, B.t()))

        loss_combined += (similarity_matrix[(cpc_mask==2)&A_filter_map].mean() + similarity_matrix[cpc_mask==3].mean())/2 - similarity_matrix[cpc_mask==1].mean() 

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += (similarity_matrix[(cpc_mask==2)&(similarity_matrix<0.999)].mean() + similarity_matrix[cpc_mask==3].mean())/2 - similarity_matrix[cpc_mask==1].mean() 

    elif hyp['optimizer']['losses']['glimpse_loss'] == 2:

        A = actvs[:,1:,:].reshape(-1, outputs[0].shape[2])
        B = outputs[0].reshape(-1, outputs[0].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T
        # MSE: similarity_matrix = -torch.sqrt(torch.sum(A**2, dim=1, keepdim=True) + torch.sum(B**2, dim=1).unsqueeze(0) - 2 * torch.mm(A, B.t()))

        loss_combined += -similarity_matrix[cpc_mask==1].mean() + similarity_matrix[(cpc_mask == 2) | (cpc_mask == 3)].mean()

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += -similarity_matrix[cpc_mask==1].mean() + similarity_matrix[(cpc_mask == 2) | (cpc_mask == 3)].mean()

    if hyp['optimizer']['losses']['semantic_loss']:

        A = semantic_embed.unsqueeze(1).repeat(1, outputs[1].shape[1], 1).reshape(-1, outputs[1].shape[2])
        B = outputs[1].reshape(-1, outputs[1].shape[2])

        A = A / A.norm(dim=1, keepdim=True)
        similarity_matrix = A @ (B / B.norm(dim=1, keepdim=True)).T

        if hyp['optimizer']['losses']['semantic_loss'] == 1: # flat mean-difference (published)

            loss_combined += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean()

            if compute_contrastive_floor:

                oracle = A @ A.T
                contrastive_loss_floor += oracle[cpc_mask==3].mean() - oracle[cpc_mask==1].mean()

        elif hyp['optimizer']['losses']['semantic_loss'] == 2: # temperature-scaled InfoNCE

            temperature = hyp['optimizer']['losses']['semantic_temp']
            norm = hyp['optimizer']['losses']['semantic_norm']
            # drop duplicate caption embeddings (several scanpaths per scene in a batch)
            same_img = None
            if hyp['optimizer']['losses']['semantic_dedup'] and img_n is not None:
                ids = img_n.reshape(-1).repeat_interleave(outputs[1].shape[1])
                same_img = ids.unsqueeze(1) == ids.unsqueeze(0)

            sem_loss, sem_diag = semantic_infonce(similarity_matrix, cpc_mask, temperature,
                                                  norm=norm, same_img=same_img)
            loss_combined += sem_loss
            diagnostics.update(sem_diag)

            if compute_contrastive_floor:

                # the oracle: prediction == the true caption embedding. Under a sharpened
                # loss this floor is informative rather than ~0 -- it is set by how
                # confusable the caption embeddings themselves are at this temperature.
                floor, floor_diag = semantic_infonce(A @ A.T, cpc_mask, temperature,
                                                     norm=norm, same_img=same_img)
                contrastive_loss_floor += floor
                diagnostics['sem/floor_nats'] = floor_diag['sem/nats']

    if hyp['optimizer']['losses']['scene_loss'] == 1:

        A = scene_embed.unsqueeze(1).repeat(1, outputs[2].shape[1], 1).reshape(-1, outputs[2].shape[2])
        B = outputs[2].reshape(-1, outputs[2].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T

        loss_combined += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean()

        if compute_contrastive_floor:

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==1].mean()

    elif hyp['optimizer']['losses']['scene_loss'] == 2:

        A = outputs[2].reshape(-1, outputs[2].shape[2])
        
        similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T

        loss_combined += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==2].mean()

        if compute_contrastive_floor: # crazy LLM upper bound!

            A = semantic_embed.unsqueeze(1).repeat(1, outputs[2].shape[1], 1).reshape(-1, semantic_embed.shape[1])

            similarity_matrix = (A / A.norm(dim=1, keepdim=True)) @ (A / A.norm(dim=1, keepdim=True)).T
            contrastive_loss_floor += similarity_matrix[cpc_mask==3].mean() - similarity_matrix[cpc_mask==2].mean()

    if hyp['optimizer']['losses']['gazeloc_loss'] == 1:

        if hyp['optimizer']['losses']['glimpse_loss'] == 1: # then ofc only relative coords are provided so have to predict next frame
            loss_combined += torch.mean((outputs[3] - fix_coords[:,1:,:])**2)**0.5
        else: # absolute coords are provided so predict them
            loss_combined += torch.mean((outputs[3] - fix_coords[:,:-1,:])**2)**0.5
    
    return loss_combined, contrastive_loss_floor, diagnostics

