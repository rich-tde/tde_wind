#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:48 2024

@author: konstantinos
"""

import argparse

def parse():
    parser = argparse.ArgumentParser(
        description="Parses simulation features"
    )

    # Add arguments to the parser
    parser.add_argument(
        "-m", "--mass",
        type = float,
        help = "Mass of the star",
        required = True
    )

    parser.add_argument(
        "-r", "--radius",
        type = float,
        help = "Radius of the star",
        required = True
    )

    parser.add_argument(
        "-b", "--blackhole",
        type = float,
        help = "Mass of the Black Hole",
        required = True
    )

    parser.add_argument(
        "-n", "--name",
        type = str,
        help = 'Name of the directory to save at',
        required = True
    )
    
    parser.add_argument(
        "-f", "--first",
        type = int,
        help = 'First snapshot to extract',
        required = True,
    )

    parser.add_argument(
        "-l", "--last",
        type = int,
        help = 'Last snapshot to extract',
        required = True,
    )

    parser.add_argument(
        "-w", "--what",
        type = str,
        help = "Quantity to project",
        required = False
    )

    parser.add_argument(
        "-ο", "--only",
        type = int,
        help = "Do something for one snapshot",
        required = False
    )
    parser.add_argument(
        "-s", "--single",
        type = bool,
        help = "Do something for one snapshot",
        required = True
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    return args
    