# textProcessing.py
    - will either iterate over all folders inside of /G8
      to generate training data for the neural network or
      it will process a text file and generate a corresponding
      csv file to throw at the neural network
    
    - training data has already been generated for the neural
      network and the NN has been trained using said data
    
    - if you wish to generate new trining material:
        - `python textProcessing.py`
            - answer yes --> `y`

    - to process a single text file:
        - `python textProcessing.py`
        - it will ask you if you want to generate new
          training material, say no --> `n`
        - enter the path to the new text file to process
            - this text file can contain anything (not poetry) written
              by either Arthur Conan Doyle, Edgar Allen Poe,
              or Henry David Thoreau.

# train.py
    - will either generate and train a new neural network
      or use a pre-trained network to identify the author
      of a given cvs file
        - csv file generated using textProcessing.py
    
    - to train a new neural network:
        - `python train.py`
        - it'll ask you if you want to train a new network,
          answer yes --> `y`
        train.py will then iterate over the two csv files
        inside of /G8. these files must be generated by
        textProcessing.py first
    
    - to make a prediction:
        - `python train.py`
        - answer no --> `n`
        - the previously saved network, enter:
            savedNetwork/savedNetwork
        - enter the relative path to the csv file
            - csv file must have first been generated
              using textProcessing.py