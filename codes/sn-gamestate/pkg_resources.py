# pkg_resources shim — setuptools 82.x removed this module; torchmetrics needs it
from importlib.metadata import PackageNotFoundError as DistributionNotFound

class _Dist:
    def __init__(self, meta):
        self._meta = meta
    def __getattr__(self, name):
        return getattr(self._meta, name)

def parse_version(v):
    from packaging.version import parse as _parse
    return _parse(str(v))

def resource_filename(package_or_requirement, resource_name):
    import importlib
    import os
    pkg = importlib.import_module(package_or_requirement)
    pkg_dir = os.path.dirname(os.path.abspath(pkg.__file__))
    return os.path.join(pkg_dir, resource_name)

def get_distribution(pkg):
    import importlib.metadata as _im
    try:
        return _Dist(_im.distribution(pkg))
    except _im.PackageNotFoundError:
        raise DistributionNotFound(pkg)

def require(reqs):
    pass

class _WorkingSet:
    def __iter__(self): return iter([])

working_set = _WorkingSet()

class VersionConflict(Exception): pass
class DistributionNotFound(Exception): pass

