from scipy._lib._public_api import PUBLIC_MODULES
from scipy._lib._array_api_docs_tables import calculate_table_statistics
from scipy._lib._array_api_docs_tables import make_flat_capabilities_table
import pandas as pd

included_modules = [
    module
    for module in PUBLIC_MODULES
    # These are extension modules which should never be included. The introspection
    # in make_flat_capabilities_table will fail for these.
    if module
    not in {
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",
        "scipy.odr",
        "scipy.fftpack",
        "scipy.stats.mstats",
        "scipy.linalg.blas",
        "scipy.linalg.lapack",
    }
]

flat_table = make_flat_capabilities_table(included_modules, "cpu")

df_cpu = pd.DataFrame.from_dict(calculate_table_statistics(flat_table), orient="index")
df_cpu.drop(columns=["dask"], inplace=True)
cols = ["torch", "jax"]
df_cpu = df_cpu[cols].div(df_cpu["total"], axis=0).multiply(100).round().astype("Int64")
df_cpu.rename(columns={col: f"{col}_cpu" for col in cols}, inplace=True)

flat_table = make_flat_capabilities_table(included_modules, "gpu")

df_gpu = pd.DataFrame.from_dict(calculate_table_statistics(flat_table), orient="index")
cols = ["cupy", "torch", "jax"]
df_gpu = df_gpu[cols].div(df_gpu["total"], axis=0).multiply(100).round().astype("Int64")
df_gpu.rename(columns={col: f"{col}_gpu" for col in cols}, inplace=True)

df_jit = pd.DataFrame.from_dict(calculate_table_statistics(flat_table), orient="index")
cols = ["jax"]
df_jit = df_jit[cols].div(df_jit["total"], axis=0).multiply(100).round().astype("Int64")
df_jit.rename(columns={col: f"{col}jit" for col in cols}, inplace=True)

df = df_cpu.join(df_gpu, how="outer").join(df_jit, how="outer")
# reorder the columns
df = df[["cupy_gpu", "torch_cpu", "torch_gpu", "jax_cpu", "jax_gpu", "jaxjit"]]

df.index = df.index.str.replace("scipy.", "", regex=False)

mean_row = df.mean(numeric_only=True).round().astype("Int64")
mean_row.name = "mean"

# add percent sign to all values
df = df.astype(str) + r" \%"
latex = df.to_latex()
# add midrule before the mean row
latex = latex.replace(
    r"\bottomrule",
    r"\midrule"
    + "\n"
    + "mean & "
    + " & ".join(mean_row.astype(str) + r" \%")
    + r" \\"
    + "\n"
    + r"\bottomrule",
)
print(latex)