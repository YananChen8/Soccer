import trackeval
print("file:", trackeval.__file__)
print("attrs:", [x for x in dir(trackeval) if not x.startswith('_')])

try:
    from trackeval.datasets import MotChallenge2DBox
    print("datasets import OK")
except Exception as e:
    print("datasets import FAILED:", e)

# Check if old trackeval is shadowing new one
import sys
for p in sys.path:
    if 'trackeval' in p.lower() or 'tracklab' in p.lower():
        print("  path:", p)
