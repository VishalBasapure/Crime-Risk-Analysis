garbled = "栩顷桥玲竺甄桥挽振枢挽攘挽挽枢挽枢纪"
raw_bytes = garbled.encode("utf-16")   # get the raw byte sequence

# Swap endianness
swapped = bytearray()
for i in range(0, len(raw_bytes), 2):
    swapped.extend(raw_bytes[i:i+2][::-1])

# Decode the corrected bytes
answer = swapped.decode("utf-16")
print(answer)