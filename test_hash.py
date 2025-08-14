import hashlib

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Test dengan data Anda
stored_hash = "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f"
password = "finance123"
generated = hash_password(password)

print(f"Password: {password}")
print(f"Generated: {generated}")
print(f"Stored:    {stored_hash}")
print(f"Match: {generated == stored_hash}")