import random 
from pinecone import Pinecone
import pinecone
import os
from glob import glob
from utility.model_utils import *

pc = Pinecone(api_key="pcsk_2Kb6sm_KqYLEeBCeeSbLV2ZngQEfJMpgiPMnjKqbBMczcPG66QeC4X8NrbngmeY2vjMndk")
index = pc.Index("criminals-index2")
    
gender=["male","female"]
felony=["Robbery","Kidnapping","Arson","Burglary","Driving under the influence","Bribery","Assault","Human trafficking","Bribery","Assault"]
c=0
for folder in glob("archive (2)/*"):
    if folder == ".DS_Store":
        continue
    if folder =="README":
        continue
    
    for path in glob(f"{folder}/*.pgm"):
        index.upsert(
            vectors=[
                (
                f"{c}",               
                get_image_embedding(path).tolist(),
                {"label": folder.split("/")[-1],"path":path,"gender":random.choice(gender),"age":random.randint(20,60),"felony":random.choice(felony)}   
                )
            ]
        )
        print(f"Vector {c} upserted successfully!")
        c=c+1
