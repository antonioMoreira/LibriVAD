## Setup LibriVAD requirements
1. Navigate to the LibriVAD directory and install python libraries

    pip install -r requirements.txt

2. Choose the mirror closer to your location from [EU, USA, CN] and run the setup.py script. EU example:

    python setup.py EU

    After downloading LibriSpeech train-clean-100, dev-clean and test-clean, the setup should take a few minutes.

## Creating LibriVAD
3. After the setup is completed, run the create_LibriVAD.py script and choose the size of the generated dataset between "small", "medium" and "large". 
   The sizes of the dataset are 15Gb, 150Gb and 1.5Tb respectively. Example:

    python create_LibriVAD.py small

The generation process could take a few hours, depending on the size choice.

!The space requirements are 61Gb for the setup, not including the size of LibriVAD (15Gb, 150Gb or 1.5Tb)


