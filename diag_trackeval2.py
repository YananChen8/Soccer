import sys
import os

# Find trackeval
for p in sys.path:
    if 'trackeval' in p.lower():
        print('sys.path:', p)

# Check vendored trackeval
vendored = '/remote-home/jiayuanrao/yishan/SoccerMaster/codes/tracklab/trackeval'
print('vendored contents:', os.listdir(vendored))

# Check if pip trackeval exists
import importlib
try:
    spec = importlib.util.find_spec('trackeval')
    print('trackeval spec:', spec)
except Exception as e:
    print('find_spec error:', e)

# Look for datasets anywhere in the env
for root, dirs, files in os.walk('/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/lib/python3.10/site-packages'):
    if 'trackeval' in root:
        print('found in site-packages:', root)
        for f in files[:20]:
            print(' ', f)
        break
