from .screen import prepare_source_matrix


def run(args):
    if args.stop_at not in (None,8192):raise ValueError('Only the fixed initial or paired 8192 matrix is supported')
    return prepare_source_matrix(args.source,extension_seed=args.seed if args.stop_at==8192 else None)
