#!/usr/bin/env python
# coding: utf-8

# Code adapted from https://stackoverflow.com/questions/45368255/error-in-loading-pickle
# original = "utils/word_data.pkl"
# destination = "utils/word_data_unix.pkl"

#import libraries
import os

def d2ux(input_file):
    """
    Converts dos linefeeds - crlf to unix - lf. "_unix" appended to the end of the destination file name after it is 
    converted to unix linefeeds.
    
    Argument(s)
    input_file: file path/name that will be converted to unix linefeeds.
    
    Returns destination file name as str.
    """
    content = ''
    outsize = 0
    # Read input file
    with open(input_file, 'rb') as infile:
        content = infile.read()
    
    # Name destination file
    name, ext = os.path.splitext(input_file)
    destination_file = "{name}_unix{ext}".format(name = name, ext=ext)
    
    # Create destination file
    with open(destination_file, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

    return destination_file
