#!/usr/bin/env python

import numpy
import argparse
import experiments
import sys
import time
from experiments.nmt import\
        RNNEncoderDecoder, \
        prototype_state,\
        sample
from experiments.nmt.sample import BeamSearch
from subprocess import Popen, PIPE

LIMIT_PROC = (60*60)*24*3  # 3 days tops
LIMIT_STEP = (60*60)*2     # report every 2 hours  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proto"   , default="prototype_search_state", help="Prototype state to use for state")
    parser.add_argument("-s", "--state"   , type=argparse.FileType('r'), help="The input state file *.pkl")
    parser.add_argument("-i", "--interval", type=int, default=LIMIT_STEP, help="Time to sleep btw reports")
    parser.add_argument("-l", "--limit"   , type=int, default=LIMIT_PROC, help="Upper limit for process")
    parser.add_argument("-v", "--val_set" , help="Validation set to calculate bleu on")
    parser.add_argument("-g", "--val_gt"  , help="Validation set gt to compare")
    return parser.parse_args()

def send_email(msg):
    import smtplib
    
    gmail_user = args.username
    gmail_pwd  = args.password
    FROM       = args.username + "@gmail.com"
    TO         = args.receiver
    SUBJECT    = "Testing sending using gmail"
    TEXT       = msg

    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        #server.sendmail(FROM, TO, message)
        server.quit()
        print 'successfully sent the mail'
    except:
        print "failed to send mail"

def main():
    
    # this loads the state specified in the prototype
    state = getattr(experiments.nmt, args.proto)()
    
    if state['bleu_script'] is None:
        sys.stderr.write('Not working')
        sys.exit()
    if args.val_set:
        state['validation_set'] = args.val_set
    if args.val_gt:
        state['validation_set_grndtruth'] = args.val_gt
    
    # make model
    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state=state, rng=rng, skip_init=True)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    
    # make beam search
    beam_search = BeamSearch(enc_dec)
    beam_search.compile()
    bleu_validator = BleuValidator(state, lm_model, beam_search, verbose=False)
    #send_email()
    
    steps = round(args.limit/args.interval)
    
    for steps in steps:
        try:
            time.sleep(args.interval)
            if bleu_validator():
                pass
            msg = ",".join(format(x, "10.4f") for x in bleu_validator.val_bleu_curve)
            
        except:
            msg = "Unexpected error:" + sys.exc_info()[0] 

        send_email(msg)

if __name__ == "__main__":
    args = parse_args()
    args.username = raw_input('Enter gmail username for sender : ')
    args.password = raw_input('Enter gmail password for sender : ')
    args.receiver = raw_input('Enter email for the recipient(s): ').split(',')
    main()
    



