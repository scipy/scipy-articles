import tempfile
import subprocess
import json
import shutil
from pathlib import Path
import argparse

from git import Repo
import pandas as pd
import matplotlib.pyplot as plt
import mpl_tectonic

mpl_tectonic.enable()
plt.style.use("scripts/scipy.mplstyle")

parser = argparse.ArgumentParser(description="Plot SLOC for SciPy versions")
parser.add_argument(
    "--all",
    action="store_true",
    help="Include all code, not just the parts SciPy maintains.",
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save the plot to the figures directory."
)
args = parser.parse_args()


columns = ["Version", "Python", "C", "C++", "Fortran 77", "Cython"]

# remove code from https://github.com/scipy/scipy/issues/21232 except the
# parts that are inactive upstream.
REMOVED_PATHS = [
    "scipy/_lib/boost",
    "scipy/_lib/boost_math",
    "scipy/spatial/qhull_src",
    "scipy/spatial/qhull",  # old location
    "scipy/fft/_pocketfft",
    "scipy/fft/_duccfft",
    "scipy/_lib/pyprima",
    "scipy/_lib/_pep440.py",
    "scipy/_lib/array_api_compat",
    "scipy/_lib/array_api_extra",
    "scipy/_lib/cobyqa",
    "scipy/optimize/_highs",
    "scipy/_lib/highs",
    "scipy/_lib/unuran",
    "scipy/optimize/_direct",
    "scipy/sparse/linalg/_dsolve/SuperLU",
    "scipy/sparse/linalg/dsolve/SuperLU",  # old location
    "scipy/optimize/_trlib",
    "scipy/_lib/_uarray",
    "scipy/_build_utils/tempita",
    "scipy/stats/libnpyrandom",
    "scipy/_lib/_docscrape.py",
]

versions = [f"1.{i}.x" for i in range(19)] + ["2.0.x"]

rows = []

with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdir = Path(tmpdirname)
    print(f"Cloning into temporary directory: {tmpdirname}")
    repo = Repo.clone_from(
        "https://github.com/scipy/scipy",
        tmpdirname,
        multi_options=["--depth=1", "--no-single-branch"],
    )
    for i, version in enumerate(versions):
        print(f"Checking out version: {version}")
        branch = f"maintenance/{version}"
        # XXX: remove this once 2.0.x is released
        if branch == "maintenance/2.0.x":
            branch = "main"

        repo.git.clean("-ffdx")
        if branch == "maintenance/1.17.x":
            shutil.rmtree(
                tmpdir / "scipy/sparse/linalg/_propack/PROPACK",
                ignore_errors=True,
            )

        repo.git.checkout(branch)
        repo.git.clean("-ffdx")
        repo.git.submodule(
            "update", "--init", "--recursive", "--jobs", "8", "--depth", "1"
        )
        if args.all:
            # At one point SciPy wrapped the entire Boost library when we really only used
            # Boost.Math so we don't count the rest of Boost in our SLOC numbers.
            boost_dir = tmpdir / "scipy" / "_lib" / "boost" / "boost"
            if boost_dir.exists():
                for path in boost_dir.iterdir():
                    if path.name != "math":
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()

            if (tmpdir / "subprojects").exists():
                dirs = ["scipy", "subprojects"]
            else:
                dirs = ["scipy"]
        else:
            for rel in REMOVED_PATHS:
                p = tmpdir / rel
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()

            dirs = ["scipy"]
            if (tmpdir / "subprojects" / "xsf").exists():
                # we want to include xsf since we maintain it
                dirs.append("subprojects/xsf")

        data = subprocess.run(
            ["tokei"] + dirs + ["-o", "json"],
            cwd=tmpdirname,
            capture_output=True,
            text=True,
        )
        json_data = json.loads(data.stdout)
        rows.append(
            [
                version,
                json_data.get("Python", {}).get("code", 0),
                json_data.get("C", {}).get("code", 0)
                + json_data.get("C Header", {}).get("code", 0),
                json_data.get("C++", {}).get("code", 0)
                + json_data.get("C++ Header", {}).get("code", 0),
                json_data.get("FORTRAN Legacy", {}).get("code", 0),
                json_data.get("Cython", {}).get("code", 0),
            ]
        )

sloc_data = pd.DataFrame(rows, columns=columns)

fig, ax = plt.subplots(figsize=(4.5, 2.5), layout="constrained")
ax.stackplot(
    range(len(sloc_data["Version"])),
    sloc_data["Python"],
    sloc_data["Cython"],
    sloc_data["Fortran 77"],
    sloc_data["C"],
    sloc_data["C++"],
    labels=["Python", "Cython", "Fortran 77", "C", "C++"],
    colors=["#3572A5", "#FEDF5B", "#8478DF", "#555555", "#E2557D"],
    rasterized=True
)
ax.set_xlabel("SciPy Version")
ax.set_ylabel("SLOC")
ax.set_xticks(range(len(sloc_data["Version"])))
ax.set_xlim(0, len(sloc_data["Version"]) - 1)
ax.set_xticklabels(s.rstrip(".x") for s in sloc_data["Version"])
fig.legend(ncols=5, loc="outside upper center")
plt.xticks(rotation=45)

if args.save:
    figures_dir = Path(__file__).parent.parent / "src" / "figures"
    name = "sloc_all.pdf" if args.all else "sloc.pdf"
    plt.savefig(figures_dir / name, dpi=600)

plt.show()
