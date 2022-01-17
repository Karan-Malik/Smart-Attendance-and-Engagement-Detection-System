import os
import capture_image
import train_image
import recognize
import detect_drowsiness
import combined
import warnings
warnings.filterwarnings("ignore")

def title_bar():
    os.system('cls')
    print("\n\n\t***** Smart Attendance and Engagement Detection System *****")
    print("\t\t\tCreated by: Karan Malik and Rigved Alankar")


def mainMenu():
    title_bar()
    print()
    print(10 * "*", "MENU", 10 * "*")
    print("[1] Register")
    print("[2] Join a Class")
    print("[3] Train Images")
    print("[4] Recognize and Attendance")
    print("[5] Drowsiness Detection")
    print("[6] Quit")
    while True:
        try:
            choice = int(input("Enter Choice: "))
            if choice == 1:
                captureFaces()
                trainImages()
                break
            elif choice == 2:
                recognizeAndDrowsy()
                break
            elif choice == 3:
                trainImages()
                break
            elif choice == 4:
                recognizeFaces()
                break
            elif choice == 5:
                drowsinessDetect()
                break
            elif choice == 6:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-6")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-6\n Try Again")
    exit



def captureFaces():
    capture_image.takeImages()
    print('\nStudent registered\n')
    

def trainImages():
    train_image.TrainImages()
    print('**Model retrained**')
    key = input("Enter any key to return main menu")
    mainMenu()

def recognizeFaces():
    recognize.recognize_attendence()
    key = input("Enter any key to return main menu")
    mainMenu()

def drowsinessDetect():
    detect_drowsiness.detect_drowsy()
    key = input("Enter any key to return main menu")
    mainMenu()

def recognizeAndDrowsy():
    combined.attendanceAndDrowsy()
    key = input("Enter any key to return main menu")
    mainMenu()


mainMenu()
