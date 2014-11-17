#!/usr/bin/env python

import numpy
import argparse
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType('r'), nargs="+", help="The input files")
    parser.add_argument("-d", "--dictionary", default='vocab.pkl', help="the name of the pickled binarized text file")
    parser.add_argument("-v", "--vocab", type=int, metavar="N",
                        help="limit vocabulary size to this number, which must "
                          "include BOS/EOS and OOV markers")
    parser.add_argument("-p", "--pickle", action="store_true",
                        help="pickle the text as a list of lists of ints")
    parser.add_argument("-c", "--count", action="store_true",
                        help="save the word counts")
    return parser.parse_args()

def main():
    sys.exit('Error: not implemented!') 

if __name__ == "__main__":
    args = parse_args()
    main()
    



