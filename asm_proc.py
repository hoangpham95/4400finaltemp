import re
import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from os.path import join

SECTION_PREF = [
    'HEADER:', '.text:', '.Pav:', '.idata', '.data', '.rdata', '.bss', '.edata:',
    '.rsrc:', '.tls', '.reloc:'
]

OP_INSTR = [
    'jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add',
    'imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb'
]

KEY = ['.dll', 'std::', ':dword']

# important keyword to interact with stack and manipulate memories
# memcpy_s and memmove_s usually comes after call
MEM_KW = ['FUNCTION', 'call']

log_train_file = open("data/log.txt", "w+")
log_test_file = open("data/log_test.txt", "w+")


class ASMProcessor:
    def __init__(self, root_dir, hashes, labels, train=True, top_features=[], min_app=30):
        self.root_dir = root_dir
        self.hashes = hashes
        self.labels = labels
        self.min_app = min_app

        self.train = train
        self.top_features = top_features
        self.log_file = log_train_file if self.train else log_test_file

    def log(self, command):
        print(command)
        self.log_file.write(command + "\n")

    def gen_token_asm(self, file_name):
        asm = join(self.root_dir, file_name + ".asm")

        with open(asm, 'r', encoding='ISO-8859-1') as asmFile:
            tokens = re.sub(r'\t|\n|\r', ' ', asmFile.read()).split()

            filtered = []
            opList = []

            for i in range(1, len(tokens) - 1):
                if tokens[i] in OP_INSTR:
                    filtered.append(tokens[i])
                    opList.append(tokens[i])

                filtered += [p for p in SECTION_PREF if p in tokens[i]]
                filtered += [k for k in KEY if k in tokens[i]]
                filtered += [tokens[i] + ' ' + tokens[i + 1]
                             for k in MEM_KW if k == tokens[i]]

                # memory and function call
                if tokens[i] == '__stdcall':
                    bigram = tokens[i] + ' ' + tokens[i + 1].partition('(')[0]
                    filtered.append(bigram)
                    filtered.append(tokens[i - 1])

                # define bytes
                if tokens[i] == 'db' and tokens[i + 1][0] == "'":
                    filtered.append(tokens[i] + ' ' + tokens[i + 1])

            counts = {}
            for i in filtered:
                counts[i] = counts.get(i, 0) + 1
            asmFile.close()
            return counts

    def gen_bag_of_words(self, tok_map, glob_dict):
        ret = {}
        for w in glob_dict:
            ret[w] = tok_map.get(w, 0)
        return ret

    def gen_file_size(self, file):
        asm = join(self.root_dir, file + ".asm")
        b = join(self.root_dir, file + ".bytes")
        return {"asm": os.stat(asm).st_size, "bytes": os.stat(b).st_size}

    def process(self):
        glob_dict = {}
        asm_map = {}
        self.log("[+] Start processing {} hashes".format(len(self.hashes)))
        glob_start = time.time()

        for f in self.hashes:
            start = time.time()
            self.log(" [-] Processing tokens for {}".format(f))
            freq = self.gen_token_asm(f)
            asm_map[f] = freq
            for tok in freq:
                glob_dict[tok] = glob_dict.get(tok, 0) + freq[tok]
            t = (time.time() - start)/60
            self.log(
                " [-] Finished processing tokens for {} in {} minutes".format(f, t))
            self.log("------------------------------------------------------")

        self.log("\n")
        glob_dict = {k: v for (k, v) in glob_dict.items() if v >= self.min_app}
        glob_features = list(glob_dict.keys())
        df = pd.DataFrame(
            columns=(['file'] + glob_features) + ['asm_sz', 'byte_sz'])

        self.log("==========================================================")
        self.log(" [-] BAG OF WORDS GENERATOR")
        self.log("==========================================================\n\n")
        for f in self.hashes:
            start = time.time()
            self.log(" [-] Processing bag of words for {}".format(f))
            bag = self.gen_bag_of_words(asm_map[f], glob_dict)
            sz = self.gen_file_size(f)

            bag['file'] = f
            bag['asm_sz'] = sz['asm']
            bag['byte_sz'] = sz['bytes']

            t = (time.time() - start)/60
            df = df.append(bag, ignore_index=True)
            self.log(
                " [-] Finished bag of words for {} in {} minutes".format(f, t))
            self.log("------------------------------------------------------")

        df['label'] = np.array(self.labels)

        self.log(
            "[+] Done processing hashes after {} minutes".format((time.time() - glob_start) * 1.0 / 60))
        self.log_file.close()
        return df

    def process_test(self):
        df = pd.DataFrame(
            columns=(['file'] + self.top_features))
        self.log("========================================================")
        self.log("[T] Processing {} hashes".format(len(self.hashes)))
        glob_start = time.time()
        for f in self.hashes:
            start = time.time()
            freq = self.gen_token_asm(f)
            sz = self.gen_file_size(f)
            dt = {}

            for feat in self.top_features:
                if feat == 'asm_sz':
                    dt[feat] = sz['asm']
                elif feat == 'byte_sz':
                    dt[feat] = sz['bytes']
                else:
                    dt[feat] = freq.get(feat, 0)

            dt['file'] = f
            df = df.append(dt, ignore_index=True)
            self.log("[T] Processing for {} took {} minutes".format(
                f, (time.time() - start)/60))

        df['label'] = np.array(self.labels)
        self.log("[T] Total processing time for test: {} minutes".format(
            (time.time()-glob_start)/60))
        return df

    def get_top_features(self, df, num_top_features):
        sub_df = df.drop(columns=['file', 'label'])
        selector = SelectKBest(chi2, k=num_top_features).fit(
            sub_df, df['label'])
        scores = selector.scores_
        d = dict(zip(sub_df.columns, scores))
        top = sorted(sub_df.columns, key=lambda x: -
                     d[x])[:num_top_features]

        new_df = df.drop(
            columns=[x for x in sub_df.columns if x not in top])
        return new_df
