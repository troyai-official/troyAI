from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="troy-ai-dev",
    version="0.2.1",
    author="DeenySivarajah341",
    author_email="DeenySivarajah341@gmail.com",
    description="A package for evolving LLM agents through adversarial training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/troyai-official/troyAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "typing-extensions>=4.0.0",
    ],
) 