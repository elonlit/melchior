#!/usr/bin/env python
#
# RODAN
# v1.0
# (c) 2020,2021,2022 Don Neumann
#

import torch
import numpy as np
import os
import sys
import re
import argparse
import pickle
import time
import glob
from torch.utils.data import Dataset, DataLoader
from fast_ctc_decode import beam_search
from ont_fast5_api.fast5_interface import get_fast5_file
from torch.multiprocessing import Queue, Process
from model.melchior import Melchior, _load_checkpoint, _load_state_dict
from model.rodan import network
from model.GCRTcall.model import Model 
from utils.loss import med_mad 

def segment(seg, s):
    seg = np.concatenate((seg, np.zeros((-len(seg)%s))))
    nrows=((seg.size-s)//s)+1
    n=seg.strides[0]
    return np.lib.stride_tricks.as_strided(seg, shape=(nrows,s), strides=(s*n, n))

def load_model(args, device='cuda:0'):
    if args.model == "melchior":
        return load_melchior(device=device)
    elif args.model == "rodan":
        return load_rodan(device=device)
    elif args.model == "gcrtcall":
        return load_GCRTcall(device=device)
    
def load_melchior(checkpoint_path='models/melchior/epoch=0-val_loss=0.43.ckpt', device='cuda:0'):
    model = Melchior(in_chans=1, embed_dim=512, depth=12)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    if sorted(list(state_dict.keys()))[0].startswith('model'):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    if sorted(list(state_dict.keys()))[0].startswith('model'):
        state_dict = {k.replace('block.', ''): v for k, v in state_dict.items() if k.startswith('block.')}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_GCRTcall(device='cuda:0'):
    model = Model()
    checkpoint = torch.load('basecallers/GCRTcall/GCRTcall_ckpt.pt', map_location=device)

    filtered = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}

    model.load_state_dict(filtered)
    model.to(device)
    model.eval()
    return model

def load_rodan(device='cuda:0'):
    model = network()
    state_dict = torch.load('basecallers/rodan/rna.torch', map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.to(device)
    model.eval()
    return model

def mp_files(dir, queue, args):
    chunkname = []
    chunks = None
    queuechunks = None
    chunkremainder = None
    for file in glob.iglob(dir+"/**/*.fast5", recursive=True):
        f5 = get_fast5_file(file, mode="r")
        for read in f5.get_reads():
            while queue.qsize() >= 100:
                time.sleep(1)
            try:
                signal = read.get_raw_data(scale=True)
                if args.debug: print("mp_files:", file)
            except:
                continue
            signal_start = 0
            signal_end = len(signal)
            med, mad = med_mad(signal[signal_start:signal_end])
            signal = (signal[signal_start:signal_end] - med) / mad
            newchunks = segment(signal, 4096)
            if chunks is not None:
                chunks = np.concatenate((chunks, newchunks), axis=0)
                queuechunks += [read.read_id] * newchunks.shape[0]
            else:
                chunks = newchunks
                queuechunks = [read.read_id] * newchunks.shape[0]
            if chunks.shape[0] >= args.batchsize:
                for i in range(0, chunks.shape[0]//args.batchsize, args.batchsize):
                    queue.put((queuechunks[:args.batchsize], chunks[:args.batchsize]))
                    chunks = chunks[args.batchsize:]
                    queuechunks = queuechunks[args.batchsize:]
        f5.close()
    if len(queuechunks) > 0:
        if args.debug: print("queuechunks:", len(queuechunks), chunks.shape[0])
        for i in range(0, int(np.ceil(chunks.shape[0]/args.batchsize)), args.batchsize):
            start = i * args.batchsize
            end = start + args.batchsize
            if end > chunks.shape[0]: end = chunks.shape[0]
            queue.put((queuechunks[start:end], chunks[start:end]))
            if args.debug: print("put last chunk", chunks[start:end].shape[0])
    queue.put(("end", None))

def mp_gpu(inqueue, outqueue, args, gpu_id):
    device = f'cuda:{gpu_id}'
    model = load_model(args, device=device)
    shtensor = None
    total_time = 0
    total_calls = 0
    total_bases = 0

    while True:
        read = inqueue.get()
        file = read[0]
        if type(file) == str: 
            avg_time = total_time / total_calls if total_calls > 0 else 0
            bases_per_second = total_bases / total_time if total_time > 0 else 0
            outqueue.put(("stats", (avg_time, bases_per_second, total_bases, total_time)))
            outqueue.put(("end", None))
            break

        chunks = read[1]
        for i in range(0, chunks.shape[0], args.batchsize):
            end = i + args.batchsize
            if end > chunks.shape[0]: end = chunks.shape[0]
            event = torch.unsqueeze(torch.FloatTensor(chunks[i:end]), 1).to(device, non_blocking=True)
            
            start_time = time.perf_counter()
            with torch.no_grad():
                out = model(event)
            end_time = time.perf_counter()
            
            call_time = end_time - start_time
            total_time += call_time
            total_calls += 1
            total_bases += out.shape[1] * out.shape[0]

            if shtensor is None:
                shtensor = torch.empty(out.shape, pin_memory=True, dtype=out.dtype)
            if out.shape[1] != shtensor.shape[1]:
                shtensor = torch.empty(out.shape, pin_memory=True, dtype=out.dtype)
            logitspre = shtensor.copy_(out).cpu().numpy()
            if args.debug: print(f"mp_gpu (GPU {gpu_id}):", logitspre.shape)
            outqueue.put((file[i:end], logitspre))
            del out
            del logitspre


def mp_write(queue, args):
    files = None
    chunks = None
    totprocessed = 0
    finish = False
    while True:
        if queue.qsize() > 0:
            newchunk = queue.get()
            if newchunk[0] == "stats":
                avg_time, bases_per_second, total_bases, total_time = newchunk[1]
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(f"Average time per basecall: {avg_time:.6f} seconds\n")
                        f.write(f"Bases per second: {bases_per_second:.2f}\n")
                        f.write(f"Total bases processed: {total_bases}\n")
                        f.write(f"Total processing time: {total_time:.2f} seconds\n")
            elif type(newchunk[0]) == str:
                if not len(files): break
                finish = True
            else:
                if chunks is not None:
                    chunks = np.concatenate((chunks, newchunk[1]), axis=1)
                    files = files + newchunk[0]
                else:
                    chunks = newchunk[1]
                    files = newchunk[0]
            while files.count(files[0]) < len(files) or finish:
                totlen = files.count(files[0])
                callchunk = chunks[:,:totlen,:]
                logits = np.transpose(np.argmax(callchunk, -1), (1, 0))
                label_blank = np.zeros((logits.shape[0], logits.shape[1] + 200))
                try:
                    out,outstr = ctcdecoder(logits, label_blank, pre=callchunk, beam_size=args.beamsize)
                except:
                    # failure in decoding
                    out = ""
                    outstr = ""
                    pass
                seq = ""
                if len(out) != len(outstr):
                    sys.stderr.write("FAIL:", len(out), len(outstr), files[0])
                    sys.exit(1)
                for j in range(len(out)):
                    seq += outstr[j]
                readid = os.path.splitext(os.path.basename(files[0]))[0]
                print(">"+readid)
                if args.reverse:
                    print(seq[::-1])
                else:
                    print(seq)
                newchunks = chunks[:,totlen:,:]
                chunks = newchunks
                files = files[totlen:]
                totprocessed+=1
                if finish and not len(files): break
            if finish: break
                
vocab = { 1:"A", 2:"C", 3:"G", 4:"T" }

def ctcdecoder(logits, label, blank=False, beam_size=5, alphabet="NACGT", pre=None):
    ret = np.zeros((label.shape[0], label.shape[1]+50))
    retstr = []
    for i in range(logits.shape[0]):
        if pre is not None:
            beamcur = beam_search(torch.softmax(torch.tensor(pre[:,i,:]), dim=-1).cpu().detach().numpy(), alphabet=alphabet, beam_size=beam_size)[0]
        prev = None
        cur = []
        pos = 0
        for j in range(logits.shape[1]):
            if not blank:
                if logits[i,j] != prev:
                    prev = logits[i,j]
                    try:
                        if prev != 0:
                            ret[i, pos] = prev
                            pos+=1
                            cur.append(vocab[prev])
                    except:
                        sys.stderr.write("ctcdecoder: fail on i:", i, "pos:", pos)
            else:
                if logits[i,j] == 0: break
                ret[i, pos] = logits[i,j] # is this right?
                cur.append(vocab[logits[i,pos]])
                pos+=1
        if pre is not None:
            retstr.append(beamcur)
        else:
            retstr.append("".join(cur))
    return ret, retstr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basecall fast5 files')
    parser.add_argument("fast5dir", default=None, type=str)
    parser.add_argument("-m", "--model", default="melchior", choices=["melchior", "rodan", "gcrtcall"], help="Type of model to use")
    parser.add_argument("-r", "--reverse", default=True, action="store_true", help="reverse for RNA (default: True)")
    parser.add_argument("-b", "--batchsize", default=32, type=int, help="default: 32")
    parser.add_argument("-B", "--beamsize", default=5, type=int, help="CTC beam search size (default: 5)")
    parser.add_argument("-e", "--errors", default=False, action="store_true")
    parser.add_argument("-d", "--debug", default=False, action="store_true")
    parser.add_argument("-o", "--output", type=str, help="Output file for time statistics")
    args = parser.parse_args()

    if args.debug: print("Using sequence len:", 4096)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    call_queue = Queue()
    write_queue = Queue()
    p1 = Process(target=mp_files, args=(args.fast5dir, call_queue, args,))
    p2 = Process(target=mp_gpu, args=(call_queue, write_queue, args, 0))
    p3 = Process(target=mp_write, args=(write_queue, args,))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()