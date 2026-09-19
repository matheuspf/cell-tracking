"""Enforce fit-only ancestors inside an otherwise calibration-enabled package."""
from .provenance import validate


def validate_stages(artifacts,root,source):
    reached=validate(artifacts,root,source,calibration=artifacts[root]['kind']=='calibration')
    for key in reached:
        if artifacts[key]['kind'] in ('model','normalizer','linear','preprocessing'):
            validate(artifacts,key,source,calibration=False)
    return reached
