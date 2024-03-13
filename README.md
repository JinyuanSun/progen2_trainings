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

To convert these sequences from FASTA to CSV format, you can use the following Python script utilizing the BioPython library:

```python
from Bio import SeqIO

# Open a new CSV file to write into
with open('ABH.csv', 'w') as csv_file:
    # Write the header row
    csv_file.write('ID,Seq\n')
    # Iterate through each sequence in the FASTA file
    for record in SeqIO.parse('ABH.fasta', 'fasta'):
        sequence = str(record.seq)
        # Write the ID and sequence to the CSV
        csv_file.write(f'{record.id},{sequence}\n')
```

This script reads each sequence from a FASTA file named `ABH.fasta` and writes the sequences into a CSV file named `ABH.csv`, following the specified CSV structure.
