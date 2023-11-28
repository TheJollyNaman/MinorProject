import record
import train
import test3
import sep2

while True:
    print("enter your choice \n1. New User \n2. Existing User")
    ch = input()

    if ch=="1":
        record.face_record()
        train.train_model()
        mar = input("Want to mark your attendence ? (y\n)")
        if mar=="y" or mar=="Y":
            sep2.capture_and_save_frame()
            test3.main()
        else:
            exit()

    elif ch=="2":
        sep2.capture_and_save_frame()
        test3.main()
