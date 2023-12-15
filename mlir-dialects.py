import argparse
from threading import local
import matplotlib.pyplot as plt
import os
import subprocess
import sys

from livereload import Server
from pathlib import Path


def repo_name_from_url(git_url: str):
    """Extract the repository name from a git URL."""
    repo_name = git_url.split("/")[-1]
    assert not repo_name.endswith(".git")
    return repo_name


def clone_or_update_repo(repo_url: str, cache_dir: str):
    """Clone or update a git repository."""
    repo_name = repo_name_from_url(repo_url)
    git_url = repo_url + ".git"

    local_repo_path = os.path.join(cache_dir, repo_name)

    if os.path.isdir(local_repo_path):
        print(f"Updating repository: {repo_name}")
        subprocess.run(["git", "-C", local_repo_path, "pull"], check=True)
    else:
        print(f"Cloning repository: {repo_name}")
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, local_repo_path], check=True
        )
    return local_repo_path


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
        "https://github.com/llvm/torch-mlir",
        "https://github.com/microsoft/Accera",
        "https://github.com/openai/triton",
        "https://github.com/llvm/circt"
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
        local_repo_path = clone_or_update_repo(repo_url, cache_dir)
        for dialect in dialects:
            current_matches = count_grep_matches(local_repo_path, dialect)
            if current_matches is not None:
                matches[repo_url][dialect] = current_matches
    return matches


def generate_html(matches: dict):
    html = """
        <html>
        <head>
        <style>
        .content {
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            margin-top: 50px;
        }
        .center {
            text-align: center;
            margin-left: auto;
            margin-right: auto;
        }
        h2 {
            margin-top: 60px;
            margin-bottom: 5px;
        }
        </style>
        </head>
        <body>
        <div class="content">
        <h1 class="center">MLIR Dialect Usage Estimates</h1>
        <div>
        This page shows estimates for the usage of MLIR dialect operations in various repositories.<br>
        <br>
        Zero counts are hidden from the plots.
        </div>
        """
    for repo_url in matches:
        repo_name = repo_name_from_url(repo_url)
        print(f"Generating plot for: {repo_name}")
        repo_matches = matches[repo_url]
        sorted_matches = sorted(repo_matches.items(), key=lambda x: x[1], reverse=True)
        sorted_matches = [x for x in sorted_matches if x[1] > 0]

        (w, h) = (6, 6)
        fig, ax = plt.subplots(figsize=(w, h))
        fig.subplots_adjust(top=0.98)
        ax.bar([x[0] for x in sorted_matches], [x[1] for x in sorted_matches])
        plt.xticks(rotation=30, ha="right")
        fig.savefig(os.path.join("_public", f"{repo_name}.png"))
        repo_ref = f"<a href='{repo_url}'>{repo_name}</a>"
        html += f"<center><h2>{repo_ref}</span></h2></center>"
        html += f"<center><img src='{repo_name}.png' /></center>\n"
        # html += f"<code>\n{repo_matches}\n</code>\n"
    html += """
        </div>
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
