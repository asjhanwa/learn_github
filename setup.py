from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-evaluator",
    version="1.0.0",
    author="LLM Evaluator Team",
    author_email="team@llm-evaluator.com",
    description="A comprehensive package for evaluating LLM answers using Azure OpenAI and Claude Sonnet models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asjhanwa/learn_github",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords="llm evaluation azure openai claude sonnet metrics nlp ai",
    project_urls={
        "Bug Reports": "https://github.com/asjhanwa/learn_github/issues",
        "Source": "https://github.com/asjhanwa/learn_github",
        "Documentation": "https://github.com/asjhanwa/learn_github/blob/main/README.md",
    },
)