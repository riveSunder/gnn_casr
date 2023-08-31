from setuptools import setup, find_packages

setup(\
        name="casr",\
        version="0.1",\
        description="Learning and distilling cellular automata rules",\
        long_description="Learning Life-like cellular automata dynamics with graph neural networks and symbolic regression",\
        packages=find_packages(include=["casr"])\
    )

