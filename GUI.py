import tkinter as tk
from tkinter import ttk
from PIL import Image
import customtkinter as ctk
import os, cv2, time, datetime, csv
from data import PriceAction, Predictors

# Fonts
font_XS = ("Consolas", 16, "normal")
font_S = ("Consolas", 18, "normal")
font_M = ("Consolas", 25, "normal")
font_L = ("Consolas", 30, "normal")
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

# File path
path = os.path.dirname(os.path.realpath(__file__))

BTC = PriceAction(TF='1H')
btc1H = BTC.load_data()
Predictors = Predictors(btc1H)

class Frame(ctk.CTkFrame):
    ''' Customtkinter frame '''
    def __init__(self, master, text="", **kwargs):
        super().__init__(master, **kwargs)
        self.label = ctk.CTkLabel(self, text=text)
        self.label.grid(row=0, column=0, padx=20)

class MenuBar(tk.Menu):
    ''' Manu bar using OS default template '''
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)

        self.configure(background= 'blue', fg='red')
        for frame in parent.frames:
            self.add_command(label=frame.title, command=lambda: parent.show_frame(frame))

class MyEntry(ctk.CTkEntry):
    ''' Creates Entry '''
    def __init__(self, master, width = 50, height = 25, font=font_XS,  **kwargs):
        super().__init__(master,
                         width=width,
                         height=height,
                         font=font_XS,
                         border_width=0,
                         corner_radius=2, 
                         **kwargs)

class MyLabel(ctk.CTkLabel):
    ''' Creates Label '''
    def __init__(self, master, text="NA", font=font_S, **kwargs):
        super().__init__(master,text=text, font=font, **kwargs)
        
    def update(self, new_text: str, **kwargs):
        self.configure(text = new_text, **kwargs)

class MyButton(ctk.CTkButton):
    ''' Creates Simple Button '''
    def __init__(self, master, text: str, command=None, width = 50, height = 25, font=font_XS,  **kwargs):
        super().__init__(master, 
                         text=text,
                         command = command,
                         font=font, 
                         width = width, 
                         height = height,
                         border_width = 0, 
                         border_color='black',
                         corner_radius=5, 
                         **kwargs)
    
    def update(self, **kwargs):
        self.configure(**kwargs)

class MySpinButton(ttk.Spinbox):
    ''' Creates Simple Button '''
    def __init__(self, master, from_ = 0, to = 10, wrap=True, width = 10, font=font_XS,  **kwargs):
        super().__init__(master, 
                         from_=from_,
                         to=to,
                         width = width, 
                         font=font, 
                         wrap=wrap,
                         **kwargs)
    
    def update(self, **kwargs):
        self.configure(**kwargs)

class FloatSpinbox(ctk.CTkFrame):
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 step_size: int | float = 1,
                 command: callable = None,
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = step_size
        self.command = command

        self.configure(fg_color=("gray78", "gray28"))  # set frame color

        self.grid_columnconfigure((0, 2), weight=0)  # buttons don't expand
        self.grid_columnconfigure(1, weight=1)  # entry expands

        self.subtract_button = ctk.CTkButton(self, text="-", width=height-6, height=height-6,
                                                       command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = ctk.CTkEntry(self, width=width-(2*height), height=height-6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

        self.add_button = ctk.CTkButton(self, text="+", width=height-6, height=height-6,
                                                  command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        # default value
        self.entry.insert(0, "0.0")

    def add_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) + self.step_size
            self.entry.delete(0, "end")
            self.entry.insert(0, value)
        except ValueError:
            return

    def subtract_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) - self.step_size
            self.entry.delete(0, "end")
            self.entry.insert(0, value)
        except ValueError:
            return

    def get(self) -> float| None:
        try:
            return float(self.entry.get())
        except ValueError:
            return None

    def set(self, value: float):
        self.entry.delete(0, "end")
        self.entry.insert(0, str(float(value)))

class MyOption(ctk.CTkOptionMenu):
    ''' Create drop down list '''
    def __init__(self, master, list: list, width = 100, height = 25, font=font_XS, **kwargs):
        super().__init__(master, 
                         values=list,
                         width = width, 
                         height = height,
                         font=font, 
                         **kwargs)
    
    def update(self, **kwargs):
        self.configure(**kwargs)

def place_n_H(widgets,rely, boundary=(0,1)):
    ''' Equally distribute widgets across frame '''
    L_bound, R_bound = boundary
    segment = R_bound-L_bound
    n = len(widgets)
    for i in range(len(widgets)):
        point = segment*(i+1)/(n+1)+L_bound
        try:
            widgets[i].place(relx = point, rely=rely, anchor='center')
        except:pass

def place_n_V(widgets,relx, boundary=(0,1), anchor='w'):
    ''' Equally distribute widgets across frame '''
    L_bound, R_bound = boundary
    segment = R_bound-L_bound
    n = len(widgets)
    for i in range(len(widgets)):
        point = segment*(i+1)/(n+1)+L_bound
        try:
            widgets[i].place(relx = relx, rely=point, anchor=anchor)
        except:pass

def add_image(frame, file_name, relx, rely, size = (200,40), anchor='center'):
    photo = ctk.CTkImage(Image.open(os.path.join(path, 'images\\', file_name), "r"), size=size)
    label = ctk.CTkLabel(frame, image = photo, text="")
    label.image = photo
    label.place(relx = relx, rely = rely, anchor = anchor)

class Main(ctk.CTk):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        ctk.CTk.wm_title(self, "Paper Trading")
        self.geometry("1100x700") 
        self.minsize(700,400) 
        container = ctk.CTkFrame(self, height=700, width=1200)
        container.pack(side="top", fill = "both", expand = "true")
        container.grid_rowconfigure(0, weight= 1)
        container.grid_columnconfigure(0, weight= 1)

        self.frames = {}

        for F in (P_Home, ''):
            try:
                frame = F(container, self)
                self.frames[F] = frame
                frame.grid(row=0, column=0, sticky="nsew")
            except:
                pass

        self.show_frame(P_Home)
        menubar = MenuBar(self)
        ctk.CTk.config(self, menu=menubar)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        self.visible_frame = cont

    def Quit_application(self):
        self.destroy()

class P_Home(ctk.CTkFrame):
    title = 'Home'
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        title = ctk.CTkLabel(self, text = "BTC predictions", font=font_L)
        title.place(relx = 0.5, rely = 0.03, anchor = 'center')
        
        subtitle = ctk.CTkLabel(self, text = "select predictors and optimise backetesting parameters to increase precision", font=font_M)
        subtitle.place(relx = 0.5, rely = 0.07, anchor = 'center')
        frame1 = Frame(self, "predictors")
        frame1.place(relx= 0.05, rely = 0.55, relwidth=0.4, relheight=0.8, anchor = 'w')

        frame2 = Frame(frame1, "backtest parameters")
        frame2.place(relx= 0.95, rely = 0.1, relwidth=0.4, relheight=0.3, anchor = 'ne')

        frame3 = Frame(self, "Result")
        frame3.place(relx = 0.95, rely = 0.55,relwidth=0.4, relheight=0.8, anchor = 'e')

        btn_exit = MyButton(self, text="Exit", command=lambda: controller.Quit_application())
        btn_exit.place(relx = 0.5, rely = 0.95, anchor = 'center')

        # modelling parameters 
        predictors = Predictors.predictors
        checkboxes = [0]*len(predictors)
        for i in range(len(predictors)):
            checkboxes[i] = ctk.CTkCheckBox(master=frame1, text=predictors[i], onvalue=1, offvalue=0)
            checkboxes[i].select()
        place_n_V(checkboxes, relx = 0.15, boundary = (0.1,0.9))

        lbl_start = MyLabel(frame2, text="start")
        ent_start = MyEntry(frame2)
        ent_start.insert(0, "9000")
        
        lbl_step = MyLabel(frame2, text="step")
        ent_step = MyEntry(frame2)
        ent_step.insert(0, "3000")
        place_n_H([lbl_start,ent_start], rely=0.3)
        place_n_H([lbl_step,ent_step], rely=0.7)

        # Results of predictions
        lbls = []
        lbl_results = []
        for text in ["precision Score: ", "True Negatives:", "False positive: ", "False negatives: ", "True Positives"]:
            lbls.append(MyLabel(frame3, text = text))
            lbl_results.append(MyLabel(frame3, text = '0'))

        place_n_V(lbls, relx=0.5, boundary = (0.05,0.5), anchor = 'e')
        place_n_V(lbl_results, relx=0.53, boundary = (0.05,0.5))
        

        btn_score = MyButton(frame1, text="Score", command=lambda: predict_score())
        btn_score.place(relx = 0.5, rely = 0.9, anchor = 'center')
        def predict_score():
            selected_predictors = []
            for i in range(len(predictors)):
                if (checkboxes[i].get()):
                    selected_predictors.append(predictors[i])
            
            print(selected_predictors)
            predictions = Predictors.backtest(start = int(ent_start.get()), step = int(ent_step.get()), predictors = selected_predictors)
            values = [*Predictors.score(predictions)]
            print(values)
            for i in range(len(values)):
                lbl_results[i].configure(text = str(values[i]))

                



