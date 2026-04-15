from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
setup(
    name  = "trace-score",
    version = "0.1.1",
    author  = "Girinath V",
    author_email  = "girinathv48@gmail.com",
    description   = "Multi-turn LLM Conversation Consistency Metric",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Giri530/trace-score",
    packages  = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
        "torch>=1.11.0",
    ],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords = [
        "nlp", "llm", "evaluation", "metrics",
        "multi-turn", "consistency", "dialogue",
        "trace-score", "conversational-ai",
    ],
)
