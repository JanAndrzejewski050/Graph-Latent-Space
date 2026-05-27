# Graph Latent Project 
Advanced Data Mining Project done by Jan Andrzejewski, Kornel Orawczak, Mateusz Matyskiel 

## Collaborators info 
### UV Environment 
In order to sync to the environment in this repository type
```
uv sync 
```

In order to add a library, write
```
uv add <library_name>
```
and then push changes to github

### Google collab
To train in google collab, we have to move the dataset and code to google drive, first we need to zip the project:
```
zip -r graph_vae_project.zip data/subset/ src/
```
Then we move it google drive and in collab run:
```
from google.colab import drive
drive.mount('/content/drive')

!pip install torch_geometric rdkit

!cp /content/drive/MyDrive/graph_vae_project.zip /content/

!unzip -q /content/graph_vae_project.zip -d /content/

```
and to train and save weights:
```
import os

os.chdir('/content/src')

!python train_model_gcn1.py

!cp /content/src/model1_weights.pth /content/drive/MyDrive/
```