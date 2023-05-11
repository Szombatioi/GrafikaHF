f = open("file.txt", "r")
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
useful = 0

print("vec3 ", end="")
for line in f:
    line = line.strip()
    if not ( line.startswith("v") or line.startswith("f")):
        continue

    
    if line.startswith("v"):
        print(f"{letters[useful]}(", end="")
        line = line[3:].split("  ")
        for i in range(len(line)):
            if line[i] == "":
                continue
            if i > 0 and i < len(line):
                print(",", end="")
            print(round(float(line[i]),2), end="")
        #print(",".join(round(float(line.split("  ")))), end="")
        print("),")
        useful += 1

    elif line.startswith("f"):
        line = line[3:].split("  ")
        for i in range(len(line)):
            if line[i] == "":
                continue
            if i > 0:
                print(",", end="")
            print(letters[int(line[i])-1], end="")
        print()

f.close()
