from tkinter import *
import os
from PIL import Image, ImageDraw, ImageFont,ImageTk
from tkinter import simpledialog
import pandas as pd
import csv

IVORY = "#FFE4C0"
PINK = "#FFBBBB"
BLUE = "#BFFFF0"
GREEN = "#BFFFF0"
BLACK = "#000000"
BG_COLOR = "#21325E"
CORRECT_COLOR = "#F1D00A"
BTN_COLOR = "#F0F0F0"

mask_str = {"mask1.jpg":"착용","mask2.jpg":"착용","mask3.jpg":"착용","mask4.jpg":"착용","mask5.jpg":"착용","normal.jpg":"미착용","incorrect_mask.jpg":"잘못된 착용"}
age_str = {0:"Young",1:"Middle",2:"Old",3:"Old",4:"Old"}
gender_str = {"female":"여자","male":"남자"}
class windows_tkinter:
    def __init__(self,start_number,train_csv_path,train_check_path,data_path, image_size=200):
        self.window = Tk()
        self.start_number= start_number
        self.current_number = start_number
        self.image_size = image_size
        self.train_csv =  pd.read_csv(train_csv_path)
        self.error = False
        self.train_check_path = train_check_path
        self.data_path = data_path
    
    def clear_button_action(self):
        self.error = False
        self.error_button['bg'] = BTN_COLOR
        self.okay_button['bg'] = BTN_COLOR
        self.next_button.pack_forget()
        self.next_button['state'] = "disabled"

    def next_button_action(self):
        
        if self.error == True:
            with open(self.train_check_path,'a', newline='') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(list(self.train_csv.iloc[self.current_number])+["Error"])
                
        self.clear_button_action()
        self.current_number += 1
        path = os.path.join(self.data_path,"train/images/",self.train_csv.iloc[self.current_number][-1])
        letter_image = Image.open(path)
        letter_image = letter_image.resize((int(self.image_size*5),int(self.image_size*6.5)))
        letter_image = ImageTk.PhotoImage(letter_image)
        self.letter_image_label['image'] = letter_image
        self.letter_image_label.image=letter_image
        
        mask_name = mask_str[self.train_csv.iloc[self.current_number][-1].split("\\")[-1]]
        
        self.mask_label['text'] = mask_name
        self.age_label['text'] = age_str[self.train_csv.iloc[self.start_number][4]//30]+f" {self.train_csv.iloc[self.start_number][4]}살"
        self.gender_label['text'] = gender_str[self.train_csv.iloc[self.current_number][2]]
        self.id_label['text'] = "id : " + str(self.train_csv.iloc[self.current_number][0])
        
            
        
    

    def button_flag(self,label):
        if label == "Error":
            self.error = True
            self.error_button['bg'] = PINK
            self.okay_button['bg'] = BTN_COLOR
        else:
            self.error = False
            self.okay_button['bg'] = PINK
            self.error_button['bg'] = BTN_COLOR
        self.next_button['state'] = "normal"

    def display_window(self):
        self.window.title("마스크 데이터셋 클리닝 확인 툴")
        self.window.resizable(False,False)
        self.window.geometry(f"{self.image_size*10}x{self.image_size*10}")
        # self.window.config(padx=10,pady=10,bg=BG_COLOR)
        self.window.config(bg=BG_COLOR)

        path = os.path.join(self.data_path,"train/images/",self.train_csv.iloc[self.start_number][-1])
        letter_image = Image.open(path)
        letter_image = letter_image.resize((int(self.image_size*5),int(self.image_size*6.5)))
        letter_image = ImageTk.PhotoImage(letter_image)
        
        self.letter_image_label = Label(self.window,image=letter_image)
        self.letter_image_label.image=letter_image
        self.letter_image_label.place(relx=0.025,rely=0.21)
        
        self.next_button = Button(self.window,text=f"NEXT",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda:self.next_button_action())
        self.next_button.place(relx=0.025,rely=0.022,relwidth=0.2,relheight=0.06)
        self.next_button['state'] = "disabled"
        
        clear_button = Button(self.window,text=f"CLEAR",width = int(self.image_size*0.2),height=int(self.image_size*0.06),font=('나눔바른펜',int(self.image_size*0.1)),bg=BTN_COLOR,command=lambda:self.clear_button_action())
        clear_button.place(relx=0.025,rely=0.12,relwidth=0.2,relheight=0.06)
        
        
        mask_name = mask_str[self.train_csv.iloc[self.start_number][-1].split("\\")[-1]]
        
        self.mask_label = Label(self.window,text=mask_name,width = int(self.image_size*0.225),height=int(self.image_size*0.1),font=('나눔바른펜',int(self.image_size*0.11)),bg=BTN_COLOR)
        self.mask_label.place(relx=0.3,rely=0.02)
        
        self.age_label = Label(self.window,text=age_str[self.train_csv.iloc[self.start_number][4]//30]+f" {self.train_csv.iloc[self.start_number][4]}살" ,width = int(self.image_size*0.225),height=int(self.image_size*0.1),font=('나눔바른펜',int(self.image_size*0.11)),bg=BTN_COLOR)
        self.age_label.place(relx=0.525,rely=0.02)
        
        self.gender_label = Label(self.window,text=gender_str[self.train_csv.iloc[self.start_number][2]],width = int(self.image_size*0.225),height=int(self.image_size*0.1),font=('나눔바른펜',int(self.image_size*0.11)),bg=BTN_COLOR)
        self.gender_label.place(relx=0.75,rely=0.02)

        
        self.okay_button = Button(self.window,text="Okay",command=lambda ccdx = "Okay":self.button_flag(ccdx))
        self.okay_button.place(relx = 0.65,rely=0.27,relheight=0.15, relwidth=0.22)
        
        self.error_button = Button(self.window,text="Error",command=lambda ccdx = "Error":self.button_flag(ccdx))
        self.error_button.place(relx = 0.65,rely=0.67,relheight=0.15, relwidth=0.22)
        
        self.id_label = Label(self.window,text="id : "+str(self.train_csv.iloc[self.start_number][0]),width = int(self.image_size*0.225),height=int(self.image_size*0.1),font=('나눔바른펜',int(self.image_size*0.11)),bg=BTN_COLOR)
        self.id_label.place(relx=0.025,rely=0.88,relheight=0.1,relwidth=0.1)

        self.window.mainloop()



if __name__ == "__main__":
    data_path = "./data"
    train_mask_path = "./data/train_mask.csv"
    
    root = Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    user_input = simpledialog.askinteger("Input", "Enter a number", parent=root)
    root.destroy()

    if user_input is not None:
        ss = windows_tkinter(start_number=0,train_csv_path=train_mask_path,train_check_path=os.path.join(data_path,"train_check.csv"),data_path = data_path,image_size=100)
        ss.display_window()
