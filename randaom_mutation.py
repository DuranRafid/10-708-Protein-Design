import requests
import re
import numpy as np
from tqdm.notebook import tqdm, trange

## PRE Defined data
RESIDUES = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def seq2matrix(seq):
    N = len(seq)
    matrix = np.zeros((20, N))
    for i in range(N):
        matrix[RESIDUES.index(seq[i]), i] = 1
    return matrix

def matrix2seq(matrix):
    return "".join(
        RESIDUES[np.argmax(matrix[:, i])] for i in range(matrix.shape[1])
    )

def randon_mutation(seq):
    matrix = seq2matrix(seq)
    i = np.random.randint(matrix.shape[1])
    matrix[:, i] = 0
    matrix[np.random.randint(20), i] = 1
    return matrix2seq(matrix)

def aromatic_mutation(seq):
    matrix = seq2matrix(seq)
    i = np.random.randint(matrix.shape[1])
    matrix[:, i] = 0
    matrix[np.random.choice([4, 18, 19]), i] = 1
    print(i,)
    return matrix2seq(matrix)


def get_fuzdrop_pllps_pdp(seq):
    session = requests.session()
    cookies = {
        "_dd_s": "logs=1&id=80ae1ab8-3f38-4300-bea8-ac47344823f7&created=1651133645675&expire=1651135587151",
        "PHPSESSID": "dgsvistkgepfsiqef8696gpft1",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:98.0) Gecko/20100101 Firefox/98.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "http://fuzpred.med.unideb.hu",
        "Connection": "close",
        "Referer": "http://fuzpred.med.unideb.hu/fuzpred/upload_fasta.php",
        "Upgrade-Insecure-Requests": "1",
    }

    url_clear = "http://fuzpred.med.unideb.hu:80/fuzpred/upload_fasta.php"
    data_clear = {"clear_fasta": "clear"}
    session.post(url_clear, headers=headers, cookies=cookies, data=data_clear)

    url_fasta = "http://fuzpred.med.unideb.hu:80/fuzpred/upload_fasta_file.php"
    data_fasta = {"sequence": seq, "add_fasta": "submit"}
    session.post(url_fasta, headers=headers, cookies=cookies, data=data_fasta)

    url_getdata = "http://fuzpred.med.unideb.hu:80/fuzpred/fuzpred_droplet.php"
    result = session.get(url_getdata, headers=headers, cookies=cookies)
    
    pllps = float(re.findall(r"LLPS\<\/sub\> = (.*)\<\/strong\>", result.text)[0])
    data = re.findall(r"dataPoints: \[(.*)\]	}];", result.text)[0]
    pdp = np.array(re.findall(r',"y":(.*?),', data), dtype=float)

    return pllps, pdp


## main function
# P04637 · P53_HUMAN as example or Neutral sample as example
seq0 = 'MSVEKMTKVEESFQKAMGLKKTIDRWRNSHTHCLWQMALGQRRNPYATLRMQDTMVQELALAKKQLLMVRQAALHQLFEKEHQQYQQELNQMGKAFYVERF'
n = 100

seqs = [seq0]
pdps = np.zeros((n, len(seq0)))
pllpss = np.zeros(n)

pllps0, pdp0 = get_fuzdrop_pllps_pdp(seq0)
pllpss = np.append(pllps0, pllpss)

for i in trange(n):
    mut_seq = aromatic_mutation(seq0) # randon_mutation or aromatic_mutation
    seqs.append(mut_seq)
    pllps, pdp = get_fuzdrop_pllps_pdp(mut_seq)
    pdps[i, :] = pdp
    pllpss[i] = pllps


## draw figures 
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])#,'no-latex'])

figure = plt.figure(figsize=(10, 4))
plt.title("Random mutation")
plt.plot(pllpss, color="black")
plt.xlabel("Number of iteration")
plt.ylabel("$p_{LLPS}$")

# heatmap for pdps
import seaborn as sns



figure = plt.figure(figsize=(20, 10))
plt.imshow(pdps, cmap="RdBu_r", vmin=0, vmax=1)
plt.axis('scaled')
plt.xlabel("Position")
plt.ylabel("Number of iteration")
plt.title("Aromatic mutation")
plt.colorbar()
