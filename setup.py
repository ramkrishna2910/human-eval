import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name="human-eval-windows",
    py_modules=["human_eval"],
    version="1.0.4",
    description="Windows-compatible fork of OpenAI's human-eval",
    author="ramkrishna2910",
    author_email="ramkrishna2910@gmail.com",
    url="https://github.com/ramkrishna2910/human-eval", 
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": [
            "evaluate_functional_correctness = human_eval.evaluate_functional_correctness:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.7",
)
