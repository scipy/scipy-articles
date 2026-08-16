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
df_cpu.rename(
    columns={col: f"{col}_cpu" for col in df_cpu.columns if col != "total"},
    inplace=True,
)
df_cpu_percent = (
    df_cpu.loc[:, df_cpu.columns != "total"]
    .div(df_cpu["total"], axis=0)
    .multiply(100)
    .round()
    .astype("Int64")
)

flat_table = make_flat_capabilities_table(included_modules, "gpu")

df_gpu = pd.DataFrame.from_dict(calculate_table_statistics(flat_table), orient="index")
df_gpu.rename(
    columns={col: f"{col}_gpu" for col in df_gpu.columns if col != "total"},
    inplace=True,
)
df_gpu_percent = (
    df_gpu.loc[:, df_gpu.columns != "total"]
    .div(df_gpu["total"], axis=0)
    .multiply(100)
    .round()
    .astype("Int64")
)

flat_table = make_flat_capabilities_table(included_modules, "jit")
df_jit = pd.DataFrame.from_dict(calculate_table_statistics(flat_table), orient="index")
df_jit.rename(
    columns={col: f"{col}_jit" for col in df_jit.columns if col != "total"},
    inplace=True,
)
df_jit_percent = (
    df_jit.loc[:, df_jit.columns != "total"]
    .div(df_jit["total"], axis=0)
    .multiply(100)
    .round()
    .astype("Int64")
)

df_percent = df_cpu_percent.join(df_gpu_percent, how="outer").join(
    df_jit_percent, how="outer"
)
df_percent = df_percent[
    ["cupy_gpu", "torch_cpu", "torch_gpu", "jax_cpu", "jax_gpu", "jax_jit"]
]
df_percent.index = df_percent.index.str.replace("scipy.", "", regex=False)
df_percent.index = [rf"\texttt{{{idx}}}" for idx in df_percent.index]

df_count = df_cpu.join(df_gpu.drop(columns="total"), how="outer").join(
    df_jit.drop(columns="total"), how="outer"
)
df_count = df_count[
    ["total", "cupy_gpu", "torch_cpu", "torch_gpu", "jax_cpu", "jax_gpu", "jax_jit"]
]
overall_count = df_count.sum(axis=0)
overall_percent = (
    overall_count.loc[overall_count.index != "total"]
    .div(overall_count["total"])
    .multiply(100)
    .round()
    .astype("Int64")
)

# add percent sign to all values
df_percent = df_percent.astype(str) + r" \%"
latex = df_percent.to_latex()
# add midrule before the mean row
latex = latex.replace(
    r"\bottomrule",
    r"\midrule"
    + "\n"
    + "Overall & "
    + " & ".join(overall_percent.astype(str) + r" \%")
    + r" \\"
    + "\n"
    + r"\bottomrule",
)
print(latex)
