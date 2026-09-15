"""One source-only invocation of the fixed complete-fork replacement ablation."""
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['44b6','6bba'], required=True)
    parser.add_argument('--arm', choices=['D10_adapted','D20_temporal'], required=True)
    args = parser.parse_args()
    from .resources import cpu_budget
    cpu_budget()
    from .source_screen import run
    run(args.source, args.arm+'_replacement')
