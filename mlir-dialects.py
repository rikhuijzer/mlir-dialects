import argparse


def count():
    print("Counting...")


def main():
    parser = argparse.ArgumentParser(description="Estimate the MLIR dialect usage.")
    parser.add_argument("mode", type=str, help="The mode to run. Can be 'count'.")
    args = parser.parse_args()

    if args.mode == "count":
        count()
    else:
        print("Unknown mode: " + args.mode)
        exit(1)


if __name__ == "__main__":
    main()
