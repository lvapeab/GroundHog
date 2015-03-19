#!/usr/bin/env python

import numpy as np
import cPickle
import sys
import pprint

if __name__ == '__main__':
    
    try:
        state_filename = sys.argv[1]
        
        state = {}
        with open(state_filename) as src:
            state.update(cPickle.load(src))
    
        pprint.pprint(state)
        
    except:
        print "Unexpected error:" + sys.exc_info()[0] 