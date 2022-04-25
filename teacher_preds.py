import numpy as np
import datasets
import models
import torch


def configure_parameters():
    lookup = {
        'feat_dim': {
            'resnet50': 2048
        },
        'expected_num_pos': {
            'pascal': 1.5,
            'coco': 2.9,
            'nuswide': 1.9,
            'cub': 31.4,
            'iNat21': 1.2  # MADE-UP NUMBER, IT'S UNKOWN YET
        },
        'linear_init_params': { # best learning rate and batch size for linear_fixed_features phase of linear_init
            'an_ls': {
                'pascal': {'linear_init_lr': 1e-4, 'linear_init_bsize': 8},
                'coco': {'linear_init_lr': 1e-4, 'linear_init_bsize': 8},
                'nuswide': {'linear_init_lr': 1e-4, 'linear_init_bsize': 16},
                'cub': {'linear_init_lr': 1e-4, 'linear_init_bsize': 8}
            },
            'role': {
                'pascal': {'linear_init_lr': 1e-3, 'linear_init_bsize': 16},
                'coco': {'linear_init_lr': 1e-3, 'linear_init_bsize': 16},
                'nuswide': {'linear_init_lr': 1e-3, 'linear_init_bsize': 16},
                'cub': {'linear_init_lr': 1e-3, 'linear_init_bsize': 8}
            }
        }
    }

    P = {}
    
    # Top-level parameters:
    P['dataset'] = 'coco' # pascal, coco, nuswide, cub, iNat21
    P['loss'] = 'em' # bce, bce_ls, iun, iu, pr, an, an_ls, wan, epr, role, em
    P['train_mode'] = 'end_to_end' # linear_fixed_features, end_to_end, linear_init
    P['val_set_variant'] = 'clean' # clean, observed
    
    # Paths and filenames:
    P['experiment_name'] = 'first-teacher'
    P['load_path'] = './data'
    P['save_path'] = './results'

    # Optimization parameters:
    if P['train_mode'] == 'linear_init':
        P['linear_init_lr'] = lookup['linear_init_params'][P['loss']][P['dataset']]['linear_init_lr']
        P['linear_init_bsize'] = lookup['linear_init_params'][P['loss']][P['dataset']]['linear_init_bsize']
    P['lr_mult'] = 10.0 # learning rate multiplier for the parameters of g
    P['stop_metric'] = 'map' # metric used to select the best epoch
    
    # Loss-specific parameters:
    P['ls_coef'] = 0.1 # label smoothing coefficient

    # Additional parameters:
    P['seed'] = 1200 # overall numpy seed
    P['use_pretrained'] = True # True, False
    P['num_workers'] = 4

    # Dataset parameters:
    P['split_seed'] = 1200 # seed for train / val splitting
    # TODO: frac below should be zero, right? iNat21 is already split
    P['val_frac'] = 0.2 # fraction of train set to split off for val
    P['ss_seed'] = 999 # seed for subsampling
    P['ss_frac_train'] = 1.0 # fraction of training set to subsample
    P['ss_frac_val'] = 1.0 # fraction of val set to subsample

    P['bias_number'] = '1'
    
    # Dependent parameters:
    if P['loss'] in ['bce', 'bce_ls']:
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'
    if P['train_mode'] == 'end_to_end':
        P['num_epochs'] = 10
        P['freeze_feature_extractor'] = False
        P['use_feats'] = False
        P['arch'] = 'resnet50'
    elif P['train_mode'] == 'linear_init':
        P['num_epochs'] = 25
        P['freeze_feature_extractor'] = True
        P['use_feats'] = True
        P['arch'] = 'linear'
    elif P['train_mode'] == 'linear_fixed_features':
        P['num_epochs'] = 25
        P['freeze_feature_extractor'] = True
        P['use_feats'] = True
        P['arch'] = 'linear'
    else:
        raise NotImplementedError('Unknown training mode.')
    P['feature_extractor_arch'] = 'resnet50'
    P['feat_dim'] = lookup['feat_dim'][P['feature_extractor_arch']]
    P['expected_num_pos'] = lookup['expected_num_pos'][P['dataset']]
    P['train_feats_file'] = './data/{}/train_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])
    P['val_feats_file'] = './data/{}/val_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])

    return P


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    P = configure_parameters()

    # set up dataloader with all training images
    ds = datasets.get_data(P)
    P['num_classes'] = ds['train'].num_classes
    print(P['num_classes'])
    imgs_to_label = torch.utils.data.DataLoader(ds['train'],
                                                batch_size=1,
                                                shuffle=False,
                                                sampler=None,
                                                num_workers=P['num_workers'],
                                                drop_last=False)

    # load model from weights
    pth = '/media/julioarroyo/aspen/single-positive-multi-label/results/multi_label_experiment_2022_04_14_20-50-02_pascal/best_model_state_f.pt'
    teacher_net = models.ImageClassifier(P)
    teacher_net.load_state_dict(torch.load(pth))

    teacher_preds = np.zeros_like(ds['train'].label_matrix)

    # for each training image, get teacher's predictions
    teacher_net.eval()
    for batch in imgs_to_label:
        # move data to GPU: 
        batch['image'] = batch['image'].to(device, non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy() # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(device, non_blocking=True)
        # forward pass: 
        with torch.set_grad_enabled(False):
            batch['logits'] = teacher_net.f(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()
            teacher_preds[batch['idx']] = batch['preds_np']

    np.save(f"data/{P['dataset']}/teacher_preds.npy", teacher_preds)
