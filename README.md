# progen2_trainings

## Training Data Format

Preferably use CSV format for the data.

Example of the desired CSV structure:

```plaintext
ID,Seq
1,MADE...
```

Protein sequences are traditionally stored in the FASTA format as plain text, like so:

```
>1
MADE
```

For example, the Mgnify cluster dataset, the size of .gz file is **78G**.
```bash
wget http://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/current_release/mgy_clusters.fa.gz && gunzip mgy_clusters.fa.gz
```

To convert these sequences from FASTA to CSV format, you can use the following Python script utilizing the BioPython library:

```python
from Bio import SeqIO

# Open a new CSV file to write into
with open('mgy_clusters.fa.csv', 'w') as csv_file:
    # Write the header row
    csv_file.write('ID,Seq\n')
    # Iterate through each sequence in the FASTA file
    for record in SeqIO.parse('mgy_clusters.fa', 'fasta'):
        sequence = str(record.seq)
        # Write the ID and sequence to the CSV
        csv_file.write(f'{record.id},{sequence}\n')
```

This script reads each sequence from a FASTA file named `mgy_clusters.fa` and writes the sequences into a CSV file named `mgy_clusters.fa.csv`, following the specified CSV structure.

### Demos

1. Training from scratch
```bash
python progen_train.py
```
This will train a tiny model from scratch using `ABH.fasta` dataset.

2. Fine-tuning ProGen2-base with LoRA
```bash
python progen_lora_tdt.py
```
This will fine-tune ProGen2-base with LoRA using `tdt.fa` dataset.