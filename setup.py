from setuptools import setup, find_packages

setup(
    name="frl-trading-framework",
    version="1.0.0",
    description="Forex Reinforcement Learning Trading Framework",
    author="FRL Team",
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "pyyaml>=6.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
)
