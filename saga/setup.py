#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

from setuptools import setup

setup(
    name="SAGA",
    version="1.0",
    description="Activity-based Multi-modal Mobility Scenario Generator for SUMO",
    url="https://github.com/lcodeca/SUMOActivityGen",
    author="Lara CODECA",
    author_email="lara.codeca@gmail.com",
    packages=["src"],
    python_requires=">=3",
    install_requires=[
        "folium",
        "lxml",
        "matplotlib",
        "numpy",
        "pyproj",
        "rtree",
        "shapely",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: EPL-2.0",
        "Programming Language :: Python :: 3",
    ],
)
