import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--variants',default='C0');p.add_argument('--workers',type=int,default=8);p.add_argument('--phase')
    a=p.parse_args()
    if a.stage=='evaluate':
        from .evaluate import run
        run(a.variants.split(','),a.workers)
    elif a.stage=='secondary':
        from .secondary import run
        run(a.phase)
    elif a.stage=='rendering_trial':
        from .rendering_trial import run
        run(a.phase)
    elif a.stage=='train':
        from .train import run
        run(a.phase)
    elif a.stage=='infer':
        from .infer import run
        run(a.phase)
    else:
        from importlib import import_module
        import_module('.'+a.stage,__package__).run()

if __name__=='__main__':main()
