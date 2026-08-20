from sn_gamestate.temporal_hrnet.temporal_nbjw import _load_adapter
import torch

base = '/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/quick_subset12'
names = ['3dcnn_k15', 'tcn_k50', 'stgcn_k50', 'transformer_k50']
for name in names:
    fname = 'kp_adapter_' + name + '.pt'
    path = base + '/' + name + '/' + fname
    a = _load_adapter(path, 'cpu')
    win = a.adapter.window_size if hasattr(a, 'adapter') else a.window_size
    print(name, type(a).__name__, 'window=' + str(win))
