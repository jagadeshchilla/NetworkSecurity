
import os
from pymongo.mongo_client import MongoClient

uri = os.getenv("MONGO_DB_URL", "mongodb://localhost:27017/test")

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)