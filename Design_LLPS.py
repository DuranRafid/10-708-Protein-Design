# %% [markdown]
# ## Design of phase-separated protein sequences using probabilistic modeling and active learning

# %% [markdown]
# 1. The propety to optimize: **LLPS propensity**
# 2. The sequence-to-function predictor： **Attention-Based NN i.e. Droppler** 
# 3. The acquisition function: **UCB (Upper uncertainty bound)**
# 4. The generative model proposing the designs or design space： **All single- or double- mutations for two typical phase separation proteins FUS (P35637) and RNA-binding protein 14 (Q96PK6)  and all possible conditions**

# %% [markdown]
# ### Produce the explict design space

# %%
import numpy as np
import pandas as pd

# %%
aa = 'ACDEFGHIKLMNPQRSTVWY'

tempurature = 37
concentration = 10
ph = 7.0
ionic = 0
crowding = 0

fus_seq = 'MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNTIFVQGLGENVTIESVADYFKQIGIIKTNKKTGQPMINLYTDRETGKLKGEATVSFDDPPSAKAAIDWFDGKEFSGNPIKVSFATRRADFNRGGGNGRGGRGRGGPMGRGGYGGGGSGGGGRGGFPSGGGGGGGQQRAGDWKCPNPTCENMNFSWRNECNQCKAPKPDGPGGGPGGSHMGGNYGDDRRGGRGGYDRGGYRGRGGDRGGFRGGRGGGDRGGFGPGKMDSRGEHRQDRRERPY'
# rbp_seq = 'MKIFVGNVDGADTTPEELAALFAPYGTVMSCAVMKQFAFVHMRENAGALRAIEALHGHELRPGRALVVEMSRPRPLNTWKIFVGNVSAACTSQELRSLFERRGRVIECDVVKDYAFVHMEKEADAKAAIAQLNGKEVKGKRINVELSTKGQKKGPGLAVQSGDKTKKPGAGDTAFPGTGGFSATFDYQQAFGNSTGGFDGQARQPTPPFFGRDRSPLRRSPPRASYVAPLTAQPATYRAQPSVSLGAAYRAQPSASLGVGYRTQPMTAQAASYRAQPSVSLGAPYRGQLASPSSQSAAASSLGPYGGAQPSASALSSYGGQAAAASSLNSYGAQGSSLASYGNQPSSYGAQAASSYGVRAAASSYNTQGAASSLGSYGAQAASYGAQSAASSLAYGAQAASYNAQPSASYNAQSAPYAAQQAASYSSQPAAYVAQPATAAAYASQPAAYAAQATTPMAGSYGAQPVVQTQLNSYGAQASMGLSGSYGAQSAAAATGSYGAAAAYGAQPSATLAAPYRTQSSASLAASYAAQQHPQAAASYRGQPGNAYDGAGQPSAAYLSMSQGAVANANSTPPPYERTRLSPPRASYDDPYKKAVAMSKRYGSDRRLAELSDYRRLSESQLSFRRSPTKSSLDYRRLPDAHSDYARYSGSYNDYLRAAQMHSGYQRRM'
# P04637 · P53_HUMAN as example or Neutral sample as example
# neu_seq =  'MSVEKMTKVEESFQKAMGLKKTIDRWRNSHTHCLWQMALGQRRNPYATLRMQDTMVQELALAKKQLLMVRQAALHQLFEKEHQQYQQELNQMGKAFYVERF'

# %% fus seq meta info and parse initial sequence
phasepred_df = pd.read_json('../data/PhaSePred_human_reviewed.json')
fus_meta = phasepred_df['P35637']
fus_interpro = fus_meta['InterPro']

zfinger_ind = np.zeros(len(fus_seq))
zfinger_ind[421:453] = 1
rrm_ind = np.zeros(len(fus_seq))
rrm_ind[286:365] = 1

charge_ind = np.array(fus_meta['Charged residue']['label'].split(','), dtype=int)
phos_ind = np.array(fus_meta['Phos']['Phos'].split(','), dtype=int)

unchangeable_ind = zfinger_ind+rrm_ind+np.abs(charge_ind)+phos_ind
unchangeable_ind = (unchangeable_ind !=0).astype(int)
changeable_ind = 1-unchangeable_ind
changeable = np.where(changeable_ind==1)[0]
# seg_ind = np.array(fus_meta[unchangeable_ind'SEG']['label'].split(','), dtype=int)

# changeable_seq = fus_seq
# changeable_seq = ''.join([np.random.choice(list(aa)) if changeable_ind[i] else fus_seq[i] for i in range(len(fus_seq))])
changeable_round = changeable

# %% [markdown]
# run the sequence-to-function predictor： **Attention-Based NN i.e. Droppler**

# %% write input file for droppler and run by cmd
import os
from tqdm.auto import trange
# os.chdir('./droppler')
df_design = pd.DataFrame(columns=['round','wt_aa','mut_pos', 'mut_aa','pLLPS'])
changeable_seq = ''.join(["N" if changeable_ind[i] else fus_seq[i] for i in range(len(fus_seq))])
changeable_round = changeable

# mut_aas = [np.random.choice(list(set(aa)-set(changeable_seq[i])),1)[0] for i in mut_poses]
for r in trange(1,101):
    df_last_round = pd.read_csv(f'../round/design_result_round_{r-1}.txt', sep='\t')
    df_last_round['mut_pos'] = df_last_round['PROTEIN NAME'].str[1:-1].astype(int)
    df_last_round_stat = df_last_round.groupby('mut_pos')['LLPS PROBABILITY'].agg(['mean', 'std'])
    df_last_round_stat['acquisition'] = df_last_round_stat['mean'] + 2*df_last_round_stat['std'] # exploration & exploitation trade-off parameter kappa=2
    pos_keep = df_last_round_stat['acquisition'].sort_values(ascending=False).index[0]
    wt_aa = df_last_round[df_last_round['mut_pos']==pos_keep].sort_values('LLPS PROBABILITY',ascending=False).iloc[0]['PROTEIN NAME'][0]
    mut_keep = df_last_round[df_last_round['mut_pos']==pos_keep].sort_values('LLPS PROBABILITY',ascending=False).iloc[0]['PROTEIN NAME'][-1]
    score_keep = df_last_round[df_last_round['mut_pos']==pos_keep].sort_values('LLPS PROBABILITY',ascending=False).iloc[0]['LLPS PROBABILITY']
    seq_keep = changeable_seq[:pos_keep] + mut_keep + changeable_seq[pos_keep+1:]
    changeable_seq = seq_keep
    changeable_round = np.delete(changeable_round, np.where(changeable_round==pos_keep)[0])

    df_round = pd.DataFrame([[r, wt_aa, pos_keep, mut_keep, score_keep,]], columns = df_design.columns)
    df_design = pd.concat([df_design,df_round])

    print(df_round)
    # new round of mutation
    with open (f'../round/design_bo_round_{r}.txt', 'w') as f:
        mut_poses = np.random.choice(changeable_round, 100)
        for mut_pos in mut_poses:
            wt_aa = changeable_seq[mut_pos]
            for i in range(20):
                mut_aa = np.random.choice(list(set(aa)-set(changeable_seq[mut_pos])),1)[0]
                mut_seq = changeable_seq[:mut_pos]+mut_aa+changeable_seq[mut_pos+1:]
                f.write(f"{wt_aa}{mut_pos}{mut_aa}\t{mut_seq}\t{tempurature}\t{concentration}\t{ph}\t{ionic}\t{crowding}\n")

    tmp = os.popen(f'python ./droppler.py -a ../round/design_bo_round_{r}.txt -o ../round/design_result_round_{r}.txt')
    tmp.read()

df_design.to_csv('../design_result.txt', sep='\t', index=False)

# %% write input file for droppler and random mutation  
df_random = pd.DataFrame(columns=['round','wt_aa','mut_pos', 'mut_aa','pLLPS'])
changeable_seq = ''.join(["N" if changeable_ind[i] else fus_seq[i] for i in range(len(fus_seq))])

pos_random = np.random.choice(len(changeable_seq), 50)
aa_random = [np.random.choice(list(set(aa)-set(changeable_seq[i])),1)[0] for i in pos_random]

with open (f'../random/random_round_50.txt', 'w') as f:
    for i in range(len(pos_random)):
        mut_pos = pos_random[i]
        wt_aa = changeable_seq[mut_pos]
        mut_aa = aa_random[i]
        mut_seq = changeable_seq[:mut_pos]+mut_aa+changeable_seq[mut_pos+1:]
        f.write(f"{wt_aa}{mut_pos}{mut_aa}\t{mut_seq}\t{tempurature}\t{concentration}\t{ph}\t{ionic}\t{crowding}\n")

tmp = os.popen(f'python ./droppler.py -a ../random/random_round_50.txt -o ../random/random_result_50.txt')
tmp.read()
# %% [markdown]
# ### Parse final seq and Analyze the results
#%% parse designed seq

original_seq = ''.join(["N" if changeable_ind[i] else fus_seq[i] for i in range(len(fus_seq))])
design_seq = original_seq
# design_seq = fus_seq
randon_seq = original_seq

df_design = pd.read_csv('../design_result.txt', sep='\t')
for i,row in df_design[['wt_aa','mut_pos','mut_aa']].iterrows():
    # if i<25:
    wt_aa, pos, mut_aa = row
    design_seq = design_seq[:pos]+mut_aa+design_seq[pos+1:]
    

df_random = pd.read_csv('../random_result_50.txt', sep='\t')
for i,row in df_random.iterrows():
    wt_aa, pos, mut_aa = row['PROTEIN NAME'][0], int(row['PROTEIN NAME'][1:-1]), row['PROTEIN NAME'][-1]
    randon_seq = randon_seq[:pos]+mut_aa+randon_seq[pos+1:]

# %% [markdown]
# ### Active learning with conditions 
from itertools import product
# os.chdir('./droppler')

tempuratures = np.arange(0, 100, 10)
concentrations = np.arange(0, 350 , 5)
phs = np.arange(3, 10, 1)
ionics = np.linspace(0, 0.01, 10)
crowdings = np.arange(1,3)

with open('../condition_exp/BO_designed_seq.txt', 'w') as f:
    for name,(t,c,p,i,cr) in enumerate(product(tempuratures, concentrations, phs, ionics, crowdings)):
        f.write(f"con_{name}\t{design_seq}\t{t}\t{c}\t{p}\t{i}\t{cr}\n")

with open('../condition_exp/random_seq.txt', 'w') as f:
    for name,(t,c,p,i,cr) in enumerate(product(tempuratures, concentrations, phs, ionics, crowdings)):
        f.write(f"cond_{name}\t{randon_seq}\t{t}\t{c}\t{p}\t{i}\t{cr}\n")

# run droppler
tmp_exp1 = os.popen(f'python ./droppler.py -a ../condition_exp/BO_designed_seq.txt -o ../condition_exp/BO_condition_result.txt')
tmp_exp1.read()

tmp_exp2 = os.popen(f'python ./droppler.py -a ../condition_exp/random_seq.txt -o ../condition_exp/random_condition_result.txt')
tmp_exp2.read()