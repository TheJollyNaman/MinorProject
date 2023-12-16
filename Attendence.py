import record
import train
import test3

while True:
    print("enter your choice \n1. New User \n2. Existing User")
    ch = input()

    if ch=="1":
        record.face_record()
        train.train_model()
        mar = input("Want to mark your attendence ? (y\n)")
        if mar=="y" or mar=="Y":
            test3.main()
        else:
            exit()

    elif ch=="2":
        test3.main()
