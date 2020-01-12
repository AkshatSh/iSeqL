import argparse


def get_args() -> argparse.ArgumentParser:
    '''
    Return CLI configuration for running server
    '''
    parser = argparse.ArgumentParser(description='Run the Active Learning Server to Support the application')
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Allow the Server to use the machines GPU'
    )
    parser.add_argument(
        '--load_state',
        action='store_true',
        help='Allow the Server to load the most recently saved state'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Launch debug version of server',
    )
    parser.add_argument(
        '--port',
        default=5000,
        type=int,
        help='the port to launch server on'
    )
    parser.add_argument(
        '--num_cpu_threads',
        default=2,
        type=int,
        help='During training, how many threads can each process use',
    )
    return parser