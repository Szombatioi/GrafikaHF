f = open("file.txt", "r")

for line in f:
    if not (line.startswith("v") or line.startswith("f")):
        continue

    if line.startswith("v"):
        line = line.strip().split("  ")[1:]
        print("(", end="")
        for i in range(len(line)):
            if i > 0 and i < len(line):
                print(",", end="")
            print(line[i], end="")
        print(")")

f.close()