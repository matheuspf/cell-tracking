import argparse

def main():
    parser = argparse.ArgumentParser(description='Offline bounded v6 execution; no downloads or legacy writes')
    parser.add_argument('stage', choices=['preflight','controls','fresh','fresh-point','repeat','image-checks','infer','report','run'])
    args, extra = parser.parse_known_args()
    if args.stage == 'infer':
        from .infer import main
        return main(extra)
    if extra: parser.error('Unexpected arguments: '+' '.join(extra))
    if args.stage in ['run','preflight']:
        from .preflight import run
        run()
    if args.stage in ['run','controls']:
        from .controls import run
        run()
    if args.stage in ['run','fresh']:
        from .fresh import run
        run()
    if args.stage in ['run','fresh-point']:
        from .fresh_point import run
        run()
    if args.stage in ['run','repeat']:
        from .repeatability import run
        run()
    if args.stage in ['run','image-checks']:
        from .image_checks import run
        run()
    if args.stage in ['run','report']:
        from .report import run
        run()

if __name__ == '__main__': main()
