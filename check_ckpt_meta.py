import torch
base = '/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/quick_subset12'
dirs = [
    ('3dcnn_k15', 'kp_adapter_3dcnn_k15.pt'),
    ('tcn_k50', 'kp_adapter_tcn_k50.pt'),
    ('stgcn_k50', 'kp_adapter_stgcn_k50.pt'),
    ('transformer_k50', 'kp_adapter_transformer_k50.pt'),
]
for d in dirs:
    path = base + '/' + d[0] + '/' + d[1]
    ck = torch.load(path, map_location='cpu')
    print('=== ' + d[0] + ' ===')
    for k, v in ck.items():
        if k == 'state_dict':
            print('  state_dict keys:', list(v.keys())[:5], '...')
        else:
            print('  ' + k + ':', v)
