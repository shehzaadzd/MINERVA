#!/usr/bin/python
#-*- coding: utf-8 -*-

import os, sys
import json
import collections
import argparse
import re
import uuid


parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str)
parser.add_argument('-basefn', type=str)
parser.add_argument('-basestr', type=str, default="python code/model/trainer.py")
parser.add_argument('-base_dir', type=str, default="/home/${USER}/logs/RL-Path-RNN")
parser.add_argument('-parseandgen', action='store_true')
parser.add_argument('-jobname', type=str, default="baseline")
args = parser.parse_args()


def load_json(fn):
    return json.load(open(fn), object_pairs_hook=collections.OrderedDict)

def add_one_item(strs, k, v):
    for i,elem in enumerate(strs):
        strs[i] += " --{} {}".format(k, v)
    return strs

def gen_from_dict(base, d):


    strs = [base]
    for k,v in d.items():
        if isinstance(v, list):
            # copy the current strs for N times
            orig_len = len(strs)
            strs *= len(v)
            for i, velem in enumerate(v):
                strs[i*orig_len: (i+1)*orig_len] = add_one_item(
                                strs[i*orig_len: (i+1)*orig_len], k, velem)
        else:
            strs = add_one_item(strs, k, v)
    return strs

def dump_out_one(base_command, id, outfn):
    in_handler = open(args.basefn)
    lines = in_handler.readlines()
    in_handler.close()
    # replace the job name
    lines = [re.sub("#SBATCH --job-name=.*", "#SBATCH --job-name={}-{:04d}".format(args.jobname, id), elem) for elem in lines]
    # #SBATCH --output=base.out
    #lines = [re.sub("#SBATCH --output=.*", "#SBATCH --output={}/{}-{:04d}.out".format(args.savefolder, args.jobname, id), elem) for elem in lines]
    out_handler = open(outfn, 'w')
    out_handler.writelines(lines)
    out_handler.write("\n")
    out_handler.write(base_command+'\n')
    out_handler.close()
    print("Finished writing to: " + outfn)

def dump_out_all(strs):
    for id,elem in enumerate(strs, 1):
#        elem += " --checkpoint_path {}/{}-{:04d}.pt".format(args.savefolder, args.jobname, id)
        if args.parseandgen:
            elem += " -gen_to {}/{}-{:04d}.gens".format(args.savefolder, args.jobname, id)
            elem += " -parse_to {}/{}-{:04d}.parses".format(args.savefolder, args.jobname, id)
        dump_out_one(elem, id, args.jobname+"_{:04d}.sh".format(id))

def main():
    fromjson = load_json(args.input)
    args.jobname = fromjson['sweeper_id']
    args.savefolder = args.base_dir+'/'+args.jobname
    fromjson = fromjson['grid']
    #add model_dir to basestr
    args.basestr += ' --base_output_dir '+args.savefolder
    strs = gen_from_dict(args.basestr, fromjson)
    dump_out_all(strs)

if __name__ == '__main__':
    main()

