"""Install before scientific imports, including spawned Python workers."""
import os
import sys

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    out = os.path.realpath(os.environ['PUBLIC946_OUT'])
    denied = ('/annotation-selection-v1/', '/strong-tracker-v2/', '/strong-tracker-v3/',
              '/image-native-tracking-v5/', '/multidata-training-v4/', '/segmentation-tracking-v6',
              '/annotations/', '/ground_truth/', '/ground-truth/', '/evaluations/')

    def audit(event, args):
        if event != 'open' or not args or isinstance(args[0], int):
            return
        try:
            path = os.path.realpath(os.fsdecode(args[0]))
        except TypeError:
            return
        own = path.startswith(out + '/')
        # Numba's compiler has a package named annotations; it contains code, not labels.
        if '/site-packages/' in path and path.endswith(('.py', '.pyc', '.so')):
            return
        if any(s in path + '/' for s in denied) and not own:
            raise PermissionError('Prediction worker cannot open evaluation/old-study paths')
        if ('.geff' in path or '/labels/' in path) and not (own and '/predictions/' in path):
            raise PermissionError('Prediction worker cannot open annotation paths')
        if own and ('/scores/' in path or '/evaluations/' in path or '/finalist_lock.json' in path):
            raise PermissionError('Prediction worker cannot open evaluator outputs')
    sys.addaudithook(audit)


def probe(annotation_path):
    try:
        open(annotation_path, 'rb')
    except PermissionError:
        return True
    raise AssertionError('Annotation guard did not deny access')
