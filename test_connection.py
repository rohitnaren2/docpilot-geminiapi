from pymongo import MongoClient
import key_param

try:
    client = MongoClient(key_param.MONGODB_URI)

    # Ping the server
    client.admin.command("ping")

    print("✅ Successfully connected to MongoDB Atlas!")

except Exception as e:
    print("❌ Connection failed")
    print(e)