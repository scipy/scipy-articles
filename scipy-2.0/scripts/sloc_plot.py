import tempfile
import subprocess
import json
import os
import shutil
from pathlib import Path

from git import Repo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("scripts/scipy.mplstyle")

columns = ["Version", "Python", "C", "C++", "Fortran 77", "Cython"]
sloc_data = pd.DataFrame(columns=columns)

versions = [f"1.{i}.x" for i in range(19)] + ["2.0.x"]

with tempfile.TemporaryDirectory() as tmpdirname:
    print(f"Cloning into temporary directory: {tmpdirname}")
    repo = Repo.clone_from("https://github.com/scipy/scipy", tmpdirname,
                           multi_options=["--depth=1", "--no-single-branch"])
    for i, version in enumerate(versions):
        print(f"Checking out version: {version}")
        branch = f"maintenance/{version}"
        # XXX: remove this once 2.0.x is released
        if branch == "maintenance/2.0.x":
            branch = "main"

        repo.git.clean("-ffdx")
        if branch == "maintenance/1.17.x":
            subprocess.run(["rm", "-rf", "scipy/sparse/linalg/_propack/PROPACK"],
                           cwd=tmpdirname)

        repo.git.checkout(branch)
        repo.git.clean("-ffdx")
        repo.git.submodule("update", "--init", "--recursive")

        # At one point SciPy wrapped the entire Boost library when we really only used
        # Boost.Math so we don't count the rest of Boost in our SLOC numbers.
        boost_dir = Path(tmpdirname) / "scipy" / "_lib" / "boost" / "boost"
        if boost_dir.exists():
            for path in boost_dir.iterdir():
                if path.name != "math":
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

        if os.path.exists(os.path.join(tmpdirname, "subprojects")):
            dirs = ["scipy", "subprojects"]
        else:
            dirs = ["scipy"]
        data = subprocess.run(
            ["tokei"] + dirs + ["-o", "json"],
            cwd=tmpdirname,
            capture_output=True,
            text=True,
        )
        json_data = json.loads(data.stdout)
        sloc_data.loc[i] = [
            version,
            json_data.get("Python", {}).get("code", 0),
            json_data.get("C", {}).get("code", 0)
            + json_data.get("C Header", {}).get("code", 0),
            json_data.get("C++", {}).get("code", 0)
            + json_data.get("C++ Header", {}).get("code", 0),
            json_data.get("FORTRAN Legacy", {}).get("code", 0),
            json_data.get("Cython", {}).get("code", 0),
        ]

fig, ax = plt.subplots(figsize=(4.5, 2.5), layout="constrained")
ax.stackplot(
    range(len(sloc_data["Version"])),
    sloc_data["Python"],
    sloc_data["Cython"],
    sloc_data["Fortran 77"],
    sloc_data["C"],
    sloc_data["C++"],
    labels=["Python", "Cython", "Fortran 77", "C", "C++"]
)
ax.set_xlabel("SciPy Version")
ax.set_ylabel("SLOC")
ax.set_xticks(range(len(sloc_data["Version"])))
ax.set_xlim(0, len(sloc_data["Version"])-1)
ax.set_xticklabels(s.rstrip(".x") for s in sloc_data["Version"])
fig.legend(ncols=5, loc="outside upper center")
plt.xticks(rotation=45)
plt.show()
