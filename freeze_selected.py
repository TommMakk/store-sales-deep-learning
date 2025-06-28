import pkg_resources

# List of packages to freeze
packages = [
    "streamlit",
    "pandas",
    "matplotlib",
    "seaborn",
    "plotly",
    "typer",
    "loguru",
    "tqdm",
    "streamlit-plotly-events"
]

with open("requirements.txt", "w") as f:
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            f.write(f"{pkg}=={version}\n")
        except pkg_resources.DistributionNotFound:
            print(f"Package {pkg} is not installed.")