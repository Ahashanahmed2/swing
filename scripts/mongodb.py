#scripts/mongodb.py
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
os.makedirs('./csv/' ,exist_ok=True)





# MongoDB-এ সংযোগ
mongourl = os.getenv("MONGO_URL")
client = MongoClient(mongourl)
db = client["candleData"]
collection = db["candledatas"]

data = list(collection.find({},{'_id':0, '__v':0}))
if data:
   df =pd.DataFrame(data)
   df.to_csv('./csv/mongodb.csv',index=False,encoding='utf-8-sig')
   client.close()
   print(f"./csv/mongodb.csv is success")

   