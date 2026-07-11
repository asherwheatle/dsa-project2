# dsa-project2

Finds peaks above a threshold for amplitude of audio data. Practically, checks if a sound will not be picked up by a microphone like Blue Yeti. You can change the threshold. 

# set up virtual environment: 

python -m venv .venv

pip install -r requirements.txt

# to get the data, run 

python datacollection.py

# to view data 

python visualization.py (you can choose the top-k peaks and the threshold when running cmd)

# to test minheap

python .\peakextraction.py --k 5 (default is 10)

