import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, PhotoImage
from PIL import Image, ImageTk
import pandas as pd
import os

# Color Definitions
BG_COLOR = "#212121"  
BTN_COLOR = "#3B3C3E"  
ACTIVE_BTN_COLOR = "#0078D7"
TEXT_COLOR = "#E7E7E8"  
# 이미지 폴더 경로
IMAGE_PATH_PREFIX = "../../data/images"  

class LabelingTool:
    def __init__(self):
        
        self.root = tk.Tk()
        self.root.title("Data Labeling Tool")
        self.root.configure(bg=BG_COLOR)
        # 창 크기가 너무 크거나 작을 경우 수정하기
        self.root.geometry("800x600")  
        

        self.data_folder_path = ""
        self.csv_file_path = ""
        self.image_label = None  # 이미지 레이블 위젯
        self.current_image = None  # 현재 표시된 이미지
        self.df = None  # CSV 데이터를 저장할 DataFrame
        self.current_index = 0  # 현재 CSV 파일에서의 인덱스
        
        self.labeled_data = []
        self.selected_gender = None
        self.selected_age = None
        self.selected_mask = None



        self.create_widgets()

    def create_widgets(self):
        
        # Open Data 버튼
        self.btn_open_data = tk.Button(self.root, text="Open Data", bg=BTN_COLOR, fg=TEXT_COLOR, command=self.open_data)
        self.btn_open_data.pack(padx=10, pady=5, fill=tk.X)

        # Open CSV 버튼
        self.btn_open_csv = tk.Button(self.root, text="Open CSV", bg=BTN_COLOR, fg=TEXT_COLOR, command=self.open_csv)
        self.btn_open_csv.pack(padx=30, pady=5, fill=tk.X)

        # 데이터 경로 표시 Entry
        self.entry_data_path = Entry(self.root, bg=BTN_COLOR, fg=TEXT_COLOR)
        self.entry_data_path.pack(fill=tk.X, padx=10, pady=5)

        # CSV 정보를 표시할 레이블 생성
        self.label_csv_info = Label(self.root, bg=BG_COLOR, fg=TEXT_COLOR)
        self.label_csv_info.pack(pady=30)

        # 이미지 표시 레이블
        self.image_label = Label(self.root)
        self.image_label.pack(side=tk.RIGHT, pady=20)

        # Prev 버튼
        self.btn_prev = tk.Button(self.root, text="Prev", bg=BTN_COLOR, fg=TEXT_COLOR)
        self.btn_prev.pack(padx=10, pady=5, side=tk.LEFT)
        self.btn_prev.bind("<ButtonPress-1>", self.on_prev_press)
        self.btn_prev.bind("<ButtonRelease-1>", self.on_button_release)

        # Next 버튼
        self.btn_next = tk.Button(self.root, text="Next", bg=BTN_COLOR, fg=TEXT_COLOR)
        self.btn_next.pack(padx=10, pady=5, side=tk.LEFT)
        self.btn_next.bind("<ButtonPress-1>", self.on_next_press)
        self.btn_next.bind("<ButtonRelease-1>", self.on_button_release)

        # 레이블링 버튼 생성
        self.create_labeling_buttons()

        # CSV 저장 버튼
        self.btn_save_csv = tk.Button(self.root, text="Save CSV", bg=BTN_COLOR, fg=TEXT_COLOR, command=self.save_csv)
        self.btn_save_csv.pack(padx=10, pady=5)

        btn_width = 100  # 버튼의 너비
        btn_height = 50  # 버튼의 높이

        btn_positions = {
            'btn_open_data': (20, 120),
            'btn_open_csv': (140, 120),
            
            'gender_buttons': {
                'female': (20, 180),
                'male': (140, 180)
            },
            'age_buttons': {
                '0-30': (20, 240),
                '30-60': (140, 240),
                '+60': (260, 240)
            },
            'mask_buttons': {
                'mask': (20, 300),
                'incorrect_mask': (140, 300),
                'normal': (260, 300)
            },
            'btn_prev': (20, 360),
            'btn_next': (140, 360),
            'btn_save_csv': (20, 420)
        }

        self.btn_open_data.place(width=btn_width, height=btn_height, x=btn_positions['btn_open_data'][0], y=btn_positions['btn_open_data'][1])
        self.btn_open_csv.place(width=btn_width, height=btn_height, x=btn_positions['btn_open_csv'][0], y=btn_positions['btn_open_csv'][1])
        self.btn_prev.place(width=btn_width, height=btn_height, x=btn_positions['btn_prev'][0], y=btn_positions['btn_prev'][1])
        self.btn_next.place(width=btn_width, height=btn_height, x=btn_positions['btn_next'][0], y=btn_positions['btn_next'][1])

        # 성별 버튼 배치
        for label, btn in self.gender_buttons.items():
            pos = btn_positions['gender_buttons'][label]
            btn.place(width=btn_width, height=btn_height, x=pos[0], y=pos[1])

        # 나이 버튼 배치
        for label, btn in self.age_buttons.items():
            pos = btn_positions['age_buttons'][label]
            btn.place(width=btn_width, height=btn_height, x=pos[0], y=pos[1])

        # 마스크 착용 버튼 배치
        for label, btn in self.mask_buttons.items():
            pos = btn_positions['mask_buttons'][label]
            btn.place(width=btn_width, height=btn_height, x=pos[0], y=pos[1])

        self.btn_save_csv.place(width=btn_width, height=btn_height, x=btn_positions['btn_save_csv'][0], y=btn_positions['btn_save_csv'][1])

        # 이미지 레이블 위치 조정
        self.image_label.place(x=380, y=100)



    def on_prev_press(self, event):
        event.widget['bg'] = ACTIVE_BTN_COLOR
        self.prev_image()

    def on_next_press(self, event):
        event.widget['bg'] = ACTIVE_BTN_COLOR
        self.next_image()

    def on_button_release(self, event):
        event.widget['bg'] = BTN_COLOR

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image_from_csv()

    def next_image(self):

        if self.selected_gender and self.selected_age and self.selected_mask:
            row = self.df.iloc[self.current_index, :5].tolist()
            image_path = self.df.iloc[self.current_index, 5]
            row.extend([image_path, self.selected_mask, self.selected_age, self.selected_gender])
            self.labeled_data.append(row)
            self.selected_gender = None
            self.selected_age = None
            self.selected_mask = None

        # 다음 이미지로 넘어가기
        if self.df is not None and self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.display_image_from_csv()

        # 레이블링 버튼 색상 초기화
        for btn_group in [self.gender_buttons, self.age_buttons, self.mask_buttons]:
            for btn in btn_group.values():
                btn['bg'] = BTN_COLOR



    def open_data(self):
        self.change_button_color(self.btn_open_data)
        self.data_folder_path = filedialog.askdirectory()
        self.entry_data_path.delete(0, tk.END)
        self.entry_data_path.insert(0, self.data_folder_path)

    def open_csv(self):
        self.change_button_color(self.btn_open_csv)
        self.csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.csv_file_path:
            self.df = pd.read_csv(self.csv_file_path, header=None)
            self.current_index = 0  # 첫 번째 이미지부터 시작
            self.display_image_from_csv()  # 첫 번째 이미지 바로 표시




    def display_image_from_csv(self):
        if self.df is not None and not self.df.empty and self.current_index < len(self.df):
            row = self.df.iloc[self.current_index]
            image_path = row[5].replace("\\", "/")  # CSV 파일의 6번째 컬럼이 이미지 경로라고 가정
            csv_info = row[:6].values  # 첫 번째부터 다섯 번째 컬럼 값

            # CSV 정보 업데이트
            self.label_csv_info.config(text=" | ".join(map(str, csv_info)))

            # 이미지 로드 및 표시
            full_image_path = os.path.join(IMAGE_PATH_PREFIX, image_path)
            image = Image.open(full_image_path)
            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_image)

    def create_labeling_buttons(self):
        
        # 성별 버튼
        self.gender_buttons = { 
        "female": tk.Button(self.root, text="여자", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('gender', 'female')),
        "male": tk.Button(self.root, text="남자", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('gender', 'male'))
        }

        # 나이 버튼
        self.age_buttons = {
            "0-30": tk.Button(self.root, text="0-30", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('age', '0-30')),
            "30-60": tk.Button(self.root, text="30-60", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('age', '30-60')),
            "+60": tk.Button(self.root, text="+60", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('age', '+60'))
        }

        # 마스크 착용 버튼
        self.mask_buttons = {
        "mask": tk.Button(self.root, text="마스크o", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('mask', 'mask')),
        "incorrect_mask": tk.Button(self.root, text="마스크△", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('mask', 'incorrect_mask')),
        "normal": tk.Button(self.root, text="마스크x", bg=BTN_COLOR, fg=TEXT_COLOR, command=lambda: self.set_label('mask', 'normal'))
    }
        
        # 버튼 패킹
        for btn in self.gender_buttons.values():
            btn.pack(side=tk.TOP, padx=5, pady=2)
        for btn in self.age_buttons.values():
            btn.pack(side=tk.TOP, padx=5, pady=2)
        for btn in self.mask_buttons.values():
            btn.pack(side=tk.TOP, padx=5, pady=2)


    def set_label(self, label_type, label_value):
    # 현재 인덱스의 이미지에 대한 레이블 설정
        if self.df is not None and not self.df.empty:
            self.df.at[self.current_index, label_type] = label_value

            # 버튼 색상 변경
            if label_type == 'gender':
                self.selected_gender = label_value
                for btn in self.gender_buttons.values():
                    btn['bg'] = BTN_COLOR
                self.gender_buttons[label_value]['bg'] = ACTIVE_BTN_COLOR
            elif label_type == 'age':
                self.selected_age = label_value
                for btn in self.age_buttons.values():
                    btn['bg'] = BTN_COLOR
                self.age_buttons[label_value]['bg'] = ACTIVE_BTN_COLOR
            elif label_type == 'mask':
                self.selected_mask = label_value
                for btn in self.mask_buttons.values():
                    btn['bg'] = BTN_COLOR
                self.mask_buttons[label_value]['bg'] = ACTIVE_BTN_COLOR
                

    def reset_label_buttons(self):
        for btn_group in [self.gender_buttons, self.age_buttons, self.mask_buttons]:
            for btn in btn_group.values():
                btn['bg'] = BTN_COLOR

    # 버튼 색상 변경 함수
    def change_button_color(self, button):
        button['bg'] = ACTIVE_BTN_COLOR

    def save_csv(self):
        # 새로운 DataFrame 생성 및 CSV 파일로 저장
        columns = self.df.columns.tolist()[:5] + ['image_path', 'mask', 'age', 'gender']
        cleaned_df = pd.DataFrame(self.labeled_data, columns=columns)
        cleaned_csv_path = os.path.join(os.path.dirname(self.csv_file_path), 'train_cleaned_hj.csv')
        cleaned_df.to_csv(cleaned_csv_path, index=False, header=None)



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = LabelingTool()
    app.run()
