"""Standard-library explicit source/target populations and artifact paths."""
from .common import WORK,DATA,Blocked,read


def clips(source,part):
    split=read(WORK/'source_partitions.json')
    if part in ('fit','calibration'):return split[source][part]
    if part=='target':
        target='6bba' if source=='44b6' else '44b6'
        return sorted(split[target]['fit']+split[target]['calibration'])
    raise Blocked('Unknown population')


def prediction(source,seed,arm,clip):return WORK/'predictions'/arm/source/str(seed)/clip


def package(source,seed,arm):return WORK/'packages'/source/str(seed)/arm


def predict(source,seed,arm,clip,part):
    if clip not in clips(source,part):raise Blocked('Prediction clip outside explicit population')
    from .inference import run
    return run(package(source,seed,arm),DATA/'train'/f'{clip}.zarr',prediction(source,seed,arm,clip),
               baseline=prediction(source,seed,'C00',clip) if arm!='C00' else None)
