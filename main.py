import sys

from workflow import run_workflow


def main(test_mode=False):
    run_workflow(test_mode=test_mode)


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
