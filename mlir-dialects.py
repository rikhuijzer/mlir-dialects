import argparse
import os
import subprocess

from pathlib import Path


def clone_or_update_repo(git_url: str, cache_dir: str):
    """Clone or update a git repository."""
    repo_name = git_url.split("/")[-1]

    assert repo_name.endswith(".git")
    repo_name = repo_name[:-4]

    local_repo_path = os.path.join(cache_dir, repo_name)

    if os.path.isdir(local_repo_path):
        print(f"Updating repository: {repo_name}")
        subprocess.run(["git", "-C", local_repo_path, "pull"], check=True)
    else:
        print(f"Cloning repository: {repo_name}")
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, local_repo_path], check=True
        )


def count_grep_matches(repo_dir: str, dialect: str):
    pattern = f"= {dialect}\\."
    args = ["rg", "-q", "--stats", pattern, "-g", "*.mlir", repo_dir]
    result = subprocess.run(args, capture_output=True, text=True)
    output = result.stdout
    for line in output.split("\n"):
        if line.endswith("matches"):
            return int(line.split()[0])


def count():
    print("Counting...")
    repositories = [
        "https://github.com/llvm/torch-mlir.git",
    ]
    dialects = [
        "affine",
        "arith",
        "tensor",
    ]
    matches = {repo: {dialect: 0 for dialect in dialects} for repo in repositories}
    cache_dir = str(Path.home() / ".cache" / "mlir-dialects")
    for repo_url in repositories:
        clone_or_update_repo(repo_url, cache_dir)
        for dialect in dialects:
            current_matches = count_grep_matches(cache_dir, dialect)
            if current_matches is not None:
                matches[repo_url][dialect] = current_matches
    print(matches)


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
