from pymongo import MongoClient

# Replace with your real connection string
MONGO_URI = "mongodb+srv://unidbb:unidbb@cluster0.c47jy6m.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)

# Choose database
db = client["unidbb"]

# Connect to collections
faces_collection = db["faces"]