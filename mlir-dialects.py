import argparse
import matplotlib.pyplot as plt
import os
import subprocess
import sys

from livereload import Server
from pathlib import Path


def repo_name_from_url(git_url: str):
    """Extract the repository name from a git URL."""
    repo_name = git_url.split("/")[-1]
    assert repo_name.endswith(".git")
    repo_name = repo_name[:-4]
    return repo_name


def clone_or_update_repo(git_url: str, cache_dir: str):
    """Clone or update a git repository."""
    repo_name = repo_name_from_url(git_url)

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
        "acc",
        "affine",
        "amdgpu",
        "amx",
        "arith",
        "arm_neon",
        "arm_sve",
        "arm_sme",
        "async",
        "bufferization",
        "cf",
        "complex",
        "dlti",
        "emitc",
        "gpu",
        "index",
        "irdl",
        "linalg",
        "llvm",
        "math",
        "memref",
        "mesh",
        "ml_program",
        "nvgpu",
        "nvvm",
        "omp",
        "pdl_interp",
        "pdl",
        "quant",
        "rocdl",
        "scf",
        "shape",
        "sparse_tensor",
        "tensor",
        "ub",
        "vector",
        "x86vector",
        "spirv",
        "tosa",
        "transform"
    ]
    matches = {repo: {dialect: 0 for dialect in dialects} for repo in repositories}
    cache_dir = str(Path.home() / ".cache" / "mlir-dialects")
    for repo_url in repositories:
        clone_or_update_repo(repo_url, cache_dir)
        for dialect in dialects:
            current_matches = count_grep_matches(cache_dir, dialect)
            if current_matches is not None:
                matches[repo_url][dialect] = current_matches
    return matches


def generate_html(matches: dict):
    html = """
        <html>
        <head>
        </head>
        <body>
        <h1>MLIR Dialect Usage Estimates</h1>
        <div>
        This page shows estimates for the usage of MLIR dialect operations in various repositories.

        Zero counts are hidden from the plots.
        </div>
        """
    for repo in matches:
        repo_name = repo_name_from_url(repo)
        repo_matches = matches[repo]
        sorted_matches = sorted(repo_matches.items(), key=lambda x: x[1], reverse=False)
        sorted_matches = [x for x in sorted_matches if x[1] > 0]

        (w, h) = (6, 6)
        fig, ax = plt.subplots(figsize=(w, h))
        fig.subplots_adjust(left=0.3)
        ax.set_title(repo_name)
        ax.barh([x[0] for x in sorted_matches], [x[1] for x in sorted_matches])
        fig.savefig(os.path.join("_public", f"{repo_name}.png"))
        html += f"<center><img src='{repo_name}.png' /></center>\n"
        # html += f"<code>\n{repo_matches}\n</code>\n"
    html += """
        </body>
        </html>
        """
    return html


def write_html():
    matches = count()
    root_dir = Path(__file__).parent.resolve()
    output_path = os.path.join(root_dir, "_public", "index.html")
    html = generate_html(matches)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print("Wrote HTML to: " + output_path)


def spawn_html_write():
    """Spawn a process to write the HTML."""
    args = ["python", "mlir-dialects.py", "html"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise Exception("Failed to spawn process.")
    for line in iter(process.stdout.readline, b''):
        sys.stdout.buffer.write(line)
        sys.stdout.flush()


def serve():
    server = Server()
    # Going via a subprocess to run the latest version of the code.
    server.watch("*.py", spawn_html_write)
    server.serve(root="_public")


def main():
    parser = argparse.ArgumentParser(description="Estimate the MLIR dialect usage.")
    parser.add_argument("mode", type=str, help="The mode to run. Can be 'count|html|serve'.")
    args = parser.parse_args()

    if args.mode == "count":
        matches = count()
        print(matches)
    elif args.mode == "html":
        write_html()
    elif args.mode == "serve":
        serve()
    else:
        print("Unknown mode: " + args.mode)
        exit(1)


if __name__ == "__main__":
    main()
