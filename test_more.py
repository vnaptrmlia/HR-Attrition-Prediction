import hashlib

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Hash yang ada di kode Anda
stored_hash = "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f"

# Test berbagai kemungkinan password
test_passwords = [
    "finance123", 
    "financial", 
    "password123", 
    "manager123", 
    "test123", 
    "finance", 
    "financial123",
    "admin123",
    "admin",
    "123456",
    "password",
    "qwerty"
]

print("🔍 TESTING VARIOUS PASSWORDS:")
print("=" * 60)

found_match = False

for pwd in test_passwords:
    generated = hash_password(pwd)
    match = generated == stored_hash
    status = "✅ MATCH!" if match else "❌ No"
    print(f"Password: {pwd:15} | {status}")
    
    if match:
        found_match = True
        print(f"🎉 FOUND! The correct password is: {pwd}")

if not found_match:
    print("\n💡 No common passwords match. Hash might be corrupted.")
    print("🔧 SOLUTION: Use the correct hash for 'finance123':")
    correct_hash = hash_password("finance123")
    print(f"Replace with: {correct_hash}")