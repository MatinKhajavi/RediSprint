import os
from dotenv import load_dotenv
import redis

load_dotenv()

# Connect to Redis
client = redis.from_url(os.environ["REDIS_URL"])

print("🔍 Checking Redis indices...\n")

try:
    # List all indices
    indices = client.execute_command("FT._LIST")
    
    if indices:
        print(f"✓ Found {len(indices)} index(es):\n")
        for idx_name in indices:
            idx_name = idx_name.decode() if isinstance(idx_name, bytes) else idx_name
            print(f"  📊 {idx_name}")
            
            # Get detailed info about each index
            try:
                info = client.execute_command("FT.INFO", idx_name)
                # Parse the info (it comes as a list of key-value pairs)
                info_dict = {}
                for i in range(0, len(info), 2):
                    key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                    val = info[i+1]
                    if isinstance(val, bytes):
                        val = val.decode()
                    info_dict[key] = val
                
                print(f"     - Documents: {info_dict.get('num_docs', 'N/A')}")
                print(f"     - Records: {info_dict.get('num_records', 'N/A')}")
                
                # Get field info
                if 'attributes' in info_dict:
                    attrs = info_dict['attributes']
                    print(f"     - Fields: {len(attrs) if isinstance(attrs, list) else 'N/A'}")
                print()
                
            except Exception as e:
                print(f"     - Could not get details: {e}\n")
    else:
        print("❌ No indices found!")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
print("📊 Redis Connection Info:")
print("="*50)
print(f"Host: {client.connection_pool.connection_kwargs.get('host')}")
print(f"Port: {client.connection_pool.connection_kwargs.get('port')}")
print(f"DB: {client.connection_pool.connection_kwargs.get('db', 0)}")
print()

# Check total keys in database
try:
    dbsize = client.dbsize()
    print(f"Total keys in database: {dbsize}")
except Exception as e:
    print(f"Could not get database size: {e}")

