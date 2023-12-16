import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import subprocess
import sys

from livereload import Server
from pathlib import Path


def repo_name_from_url(repo_url: str) -> str:
    """Extract the repository name from the subdir or the git url."""
    repo_name = repo_url.split("/")[-1]
    assert not repo_name.endswith(".git")
    return repo_name


@dataclass(frozen=True)
class Repo:
    url: str
    subdir: str

    def name(self) -> str:
        if self.subdir != "":
            return self.subdir
        else:
            return repo_name_from_url(self.url)


def clone_or_update_repo(repo: Repo, cache_dir: str, update: bool):
    """Clone or update a git repository."""
    repo_name = repo.name()
    git_url = repo.url + ".git"

    repo_dir_name = repo_name_from_url(repo.url)
    local_repo_path = os.path.join(cache_dir, repo_dir_name)

    if os.path.isdir(local_repo_path):
        print(f"Updating repository: {repo_name}")
        # Tensorflow updates are very slow, so skip them.
        if update and repo_name != "tensorflow":
            subprocess.run(["git", "-C", local_repo_path, "pull"], check=True)
    else:
        print(f"Cloning repository: {repo_name}")
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, local_repo_path], check=True
        )
    return local_repo_path


def count_grep_matches(search_dir: str, dialect: str):
    pattern = f"= \"?{dialect}\\."
    glob = "*.mlir"
    if search_dir.endswith("flang"):
        glob = "*.fir"
    args = ["rg", "-q", "--stats", pattern, "-g", glob, search_dir]
    result = subprocess.run(args, capture_output=True, text=True)
    output = result.stdout
    for line in output.split("\n"):
        if line.endswith("matched lines"):
            return int(line.split()[0])


def count(update: bool):
    print("Counting...")
    repositories = [
        Repo("https://github.com/llvm/llvm-project", "mlir"),
        Repo("https://github.com/llvm/llvm-project", "flang"),
        Repo("https://github.com/google/iree", ""),
        Repo("https://github.com/tensorflow/tensorflow", ""),
        Repo("https://github.com/openxla/xla", ""),
        Repo("https://github.com/llvm/torch-mlir", ""),
        Repo("https://github.com/openai/triton", ""),
        Repo("https://github.com/Xilinx/mlir-aie", ""),
        Repo("https://github.com/onnx/onnx-mlir", ""),
        Repo("https://github.com/llvm/Polygeist", ""),
        Repo("https://github.com/llvm/circt", ""),
        Repo("https://github.com/microsoft/Accera", ""),
    ]
    dialects = [
        "acc",
        "affine",
        "amdgpu",
        "amx",
        "arith",
        "arm_neon",
        "arm_sme",
        "arm_sve",
        "async",
        "bufferization",
        "cf",
        "chlo", # tensorflow
        "complex",
        "dlti",
        "emitc",
        "fir",  # flang
        "gpu",
        "index",
        "irdl",
        "iree_vector_ext", # iree
        "iree_input", # iree
        "linalg",
        "llvm",
        "math",
        "memref",
        "mesh",
        "mhlo", # tensorflow
        "ml_program",
        "nvgpu",
        "nvvm",
        "omp",
        "pdl",
        "pdl_interp",
        "quant",
        "rocdl",
        "scf",
        "shape",
        "sparse_tensor",
        "spirv",
        "tensor",
        "tf", # tensorflow
        "tm_tensor", # torch-mlir
        "tosa",
        "transform",
        "triton_gpu", # triton
        "triton_nvidia_gpu", # triton
        "tt", # triton
        "ub",
        "vector",
        "x86vector",
    ]
    matches = {repo: {dialect: 0 for dialect in dialects} for repo in repositories}
    cache_dir = str(Path.home() / ".cache" / "mlir-dialects")
    for i, repo in enumerate(repositories):
        local_repo_path = clone_or_update_repo(repo, cache_dir, update)
        for dialect in dialects:
            # Only update first few during development.
            if update or True: # or i < 4:
                search_dir = local_repo_path
                if repo.subdir != "":
                    search_dir = os.path.join(local_repo_path, repo.subdir)
                current_matches = count_grep_matches(search_dir, dialect)
                if current_matches is not None:
                    matches[repo][dialect] = current_matches
    return matches


def generate_html(matches: dict[Repo, dict]):
    html = """
        <html>
        <head>
        <style>
        body {
            font-size: 18px;
            line-height: 1.3;
        }
        div {
            margin: 0.6em 0;
        }
        pre {
            background-color: #f0f0f0;
            border: 1px solid #dbdbdb;
            display: block;
            padding: 0.3em;
        }
        pre, code {
            font-size: 26px;
            margin-top: 0.6em;
            margin-bottom: 0.6em;
            border-radius: 4px;
        }
        h1 {
            margin: 1.2em 0;
        }
        .content {
            width: 100%;
            margin-left: 8px;
            margin-right: 5px;
        }
        @media only screen and (min-width: 800px) {
            pre, code {
                font-size: 14px;
            }
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
                margin-top: 90px;
                margin-bottom: 5px;
            }
        }
        </style>
        </head>
        <body>
        <div class="content">
        <h1 class="center">MLIR Dialect Usage Estimates</h1>
        <div>
        <p>
        This page shows estimates for the usage of MLIR dialect operations for various repositories:<br>
        </p>
        <ul>
        """
    for repo in matches:
        repo_name = repo.name()
        html += f"<li><a href='#{repo_name}'>{repo_name}</a></li>\n"
    html += """
        </ul>
        <p>
        Usage is estimated by counting the number of matches for each dialect operation in the repository.
        This is based on the assumption that a more important dialect is mentioned more often in test and example files.
        Specifically, the following ripgrep `rg` command is used:<br>
        </p>
        <pre>rg '= "?some_dialect.' -g '*.mlir' repo_dir</pre>
        <p>
        where <code>some_dialect</code> is the dialect operation to count and <code>repo_dir</code> is the repository directory (plus sub directory for some monorepo's).
        In words, this greps the repository for the string <code>= some_dialect.</code> or <code>= "some_dialect."</code> in all <code>*.mlir</code> files.
        Finally, zero counts are hidden from the plots.
        </p>
        <p>
        The source code for this page is available at
        <a href="https://github.com/rikhuijzer/mlir-dialects">https://github.com/rikhuijzer/mlir-dialects</a>.
        </p>
        </div>
        """
    for repo in matches:
        repo_name = repo.name()
        print(f"Generating plot for: {repo_name}")
        repo_matches = matches[repo]
        sorted_matches = sorted(repo_matches.items(), key=lambda x: x[1], reverse=True)
        sorted_matches = [x for x in sorted_matches if x[1] > 0]

        (w, h) = (9, 6)
        fig, ax = plt.subplots(figsize=(w, h))
        fig.subplots_adjust(left=0.06)
        fig.subplots_adjust(right=0.88)
        fig.subplots_adjust(top=0.98)
        fig.subplots_adjust(bottom=0.18) # triton_nvidia_gpu
        labels = [x[0] for x in sorted_matches]
        valus = [x[1] for x in sorted_matches]
        ax.bar(labels, valus)
        plt.xticks(rotation=43, ha="right")
        fig.savefig(os.path.join("_public", f"{repo_name}.png"))
        html += f"<a id='{repo_name}'></a>\n"
        html += f"<center><h2>{repo_name}</span></h2></center>"
        html += f"<center><a href='{repo.url}'>{repo.url}</a></center>"
        html += f"<center><img src='{repo_name}.png' /></center>\n"
        # html += f"<code>\n{repo_matches}\n</code>\n"
    html += """
        </div>
        </body>
        </html>
        """
    return html


def write_html(update: bool):
    matches = count(update)
    root_dir = Path(__file__).parent.resolve()
    output_path = os.path.join(root_dir, "_public", "index.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = generate_html(matches)
    with open(output_path, "w") as f:
        f.write(html)
    print("Wrote HTML to: " + output_path)


def spawn_html_write(update: bool):
    """Spawn a process to write the HTML."""
    args = ["python", "mlir-dialects.py"]
    if update:
        args += ["html"]
    else:
        args += ["html-noupdate"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise Exception("Failed to spawn process.")
    for line in iter(process.stdout.readline, b""):
        sys.stdout.buffer.write(line)
        sys.stdout.flush()


def serve():
    server = Server()
    update = True
    spawn_html_write(update)
    spawn_html_no_update = lambda: spawn_html_write(False)
    # Going via a subprocess to run the latest version of the code.
    server.watch("*.py", spawn_html_no_update)
    server.serve(root="_public")


def main():
    parser = argparse.ArgumentParser(description="Estimate the MLIR dialect usage.")
    parser.add_argument(
        "mode", type=str, help="The mode to run. Can be 'count|html|serve'."
    )
    args = parser.parse_args()

    if args.mode == "count":
        matches = count(update=True)
        print(matches)
    elif args.mode == "html":
        update = True
        write_html(update)
    elif args.mode == "html-noupdate":
        update = False
        write_html(update)
    elif args.mode == "serve":
        serve()
    else:
        print("Unknown mode: " + args.mode)
        exit(1)


if __name__ == "__main__":
    main()
