from setuptools import setup, find_packages

setup(
    name="elegant_finrl",
    version="0.3.1",
    author="Xiaoyang Liu, Steven Li, Hongyang Yang, Jiahao Zheng",
    author_email="XL2427@columbia.edu",
    url="https://github.com/AI4Finance-LLC/Elegant-finrl",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        'gym', 'matplotlib', 'numpy', 'torch', 'opencv-python', 'yfinance', 'stockstats'],
    description="finrl 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Deep Reinforcment Learning",
    python_requires=">=3.6",
)
