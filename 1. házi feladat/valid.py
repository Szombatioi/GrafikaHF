while(True):
    x,y,z = input("$ ").split(",")
    x = float(x)
    y = float(y)
    z = float(z)

    print("valid" if x*x+y*y-z*z >= -1.1 and x*x+y*y-z*z <= -0.9 else "invalid")
    input()
