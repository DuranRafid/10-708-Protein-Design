# WHAT IS Droppler #

Droppler is a Neural Network based method for the prediction of protein liquid liquid phase separation given a list of experimental conditions

### What is this repository for? ###

The code here contains a standalone version of Droppler (`droppler.py`), which takes as input protein sequences and experimental conditions and outputs the predicted LLPS probability. 
Droppler can handle two types of inputs:
1) A FASTA file containing the sequences and a list of conditions passed via command line. In this case all the sequences in the FASTA file will be tested using the same experimental conditions
2) A tab separated CSV file with the followin format:
    name,seq,temp,conc,ph,salt,crowd
    ProteinName : Sequence : Temperature (°C) : Protein Concentration (&#956;M) : pH  : Ionic strength : crowding Agent (1=present, 0= absent)
Input examples are available in the `test` folder.

### DATA AVAILABILITY ###

All the datasets mentioned in the paper are publicly available from the respective papers and repositories.

### How do I set it up? ###

Droppler has some dependencies, which are popular pytohn libraries (such as pytorch, numpy and scipy). 
Here we show how to create a miniconda environment containig all those libraries. Similar instructions can be used to install the dependencies with `pip`. 
Droppler runs on python 3 .

* Download and install miniconda from `https://docs.conda.io/en/latest/miniconda.html`
* Create a new conda environment by typing: `conda create -n droppler`
* Enter the environment by typing: `conda activate droppler`
* Install pytorch >= 1.0 with the command: `conda install pytorch -c pytorch` or refer to pytorch website https://pytorch.org
* Install numpy with the command: `conda install numpy`
	
You can remove this environment at any time by typing: conda remove -n droppler --all
 

### What is this repository contains? ###

* `Droppler.py` -> is the standalone predictor.
* `SA.py` -> Contains the attention modules necessary to run Droppler.
* `sources/` -> folder containing source code necessary to run Droppler
* `test/` -> folder containing a runnable exmple of FASTA file or CSV that could be used as inputs for Droppler
* `README.md` -> this readme

### How do I predict proteins with Droppler? ###


You can run one of our examples by typing:

```
python droppler.py -fasta test/test.fasta --concentration 10 --salt 0.01 --temperature 37 -crowd 0
```

You will obtain the following results:

```
PROTEIN NAME	PS PROB
Q63HQ2   0.687
O75072   0.79
```

If you prefer to use a CSV input in which you define the experimental conditions for each protein, you can the example (2 proteins, one of which with two different experimental setups) with:
```
python standalone.py -a test/test.csv 
```
in this case you will obtain 
```
PROTEIN NAME	TEMPERATURE	CONCENTRATION	PH	IONIC STRENGTH	CROWDING AGENT	LLPS PROBABILITY
Prot1	37	10	7	0.001	True	0.225
Prot2	60	1	3	0.1	True	0.149
Prot2	37	1	3	0.1	True	0.121
```

### Droppler options ##
* `-h, --help`            show an help message and exit
* `-fasta FASTA, --fasta FASTA` the input fasta file containing the protein sequences (command line input only)
               
* `-s SALT, --salt SALT`  Ionic strength of the solution  (command line input only)
*  `-ph PH, --ph PH`       pH of the solution  (command line input only)
*  `-c CONCENTRATION, --concentration CONCENTRATION`   protein concentration (uM)  (command line input only)
*  `-t TEMPERATURE, --temperature TEMPERATURE` temperature (°C)  (command line input only)
*  `-crowd CROWDINGAGENT, --crowdingAgent CROWDINGAGENT`   1 = presence of crowding agents, 2 = absence of
                        crowding agents  (command line input only)
*  `-cuda, --cuda  ` run predictions on GPU. It requires pytorch with CUDA. It is faster but it requires a dedicated GPU
*  `-o OUTFILE, --outfile OUTFILE` redirect output to file
*  `-a COMMANDFILE, --commandFile COMMANDFILE`  
predict using an input CSV. Provide the path to the
                        CSV file. See the README for its format. ALL THE OTHER
                        CONDITIONS PARAMETERS PROVIDED IN COMMAND LINE WILL BE
                        IGNORED


### Who do I talk to? ###

Please report bugs at the following mail addresses:
gabriele DoT orlando aT kuleuven DoT VIB DoT be
daniele DoT raimondi aT kuleuven DoT be

