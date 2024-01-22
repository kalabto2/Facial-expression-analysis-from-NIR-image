from datetime import datetime
import tkinter as tk
from PIL import ImageTk,Image
import pandas as pd
class GUI:
    def __init__(self):
        self.root = None
        self.canvas = None
        self.oval = None
        self.emo = None
        self.val = None
        self.aro = None
        self.__x = 506
        self.__y = 457
        self.last_filename = None
    def move_dot(self,event):
        self.canvas.delete(self.oval)
        newX = int( self.__x + self.val.get() * 394)
        newY = int( self.__y - self.aro.get() * 394)
        self.oval = self.canvas.create_oval(newX-10,newY-10,newX+10,newY+10,outline="#ff00ff",fill="#00ff00")
    def annotate_image(self, annotImg, gender, adult, race, nth, n):#(self,annotImg="305.jpg", gender="male", adult="no", race="euro"):
        self.root = tk.Tk()
        self.root.title('GUI Annotator CZ   ' + 
                        "    %s. obrázek k anotování z %s" % (str(nth), str(n)))
        #self.root.state('zoomed')
        self.root.attributes('-fullscreen', True)
        self.canvas = tk.Canvas(self.root, width = 1680, height = 1050)  
        self.canvas.pack() 
        ###############
        # ESSENTIALS ##
        ###############
        label1 = tk.Label(self.root, text="Valence-Arousal Model - pomůcka --- !! prosím držte se v jednotkové kružnici, díky !!")
        self.canvas.create_window(300,10,window=label1)
        img = ImageTk.PhotoImage(Image.open("VAmodel.png").resize((934,868),resample=2), master = self.canvas)  
        self.canvas.create_image(35, 25, anchor=tk.NW, image=img)
        label2 = tk.Label(self.root, text=" Subjekt : %s -- %s -- %s" % (gender, ("18-" if adult=="no" else "18+"), race))
        self.canvas.create_window(1150,10,window=label2)
        label3 = tk.Label(self.root, text="Emoce:")
        self.canvas.create_window(1080,550,window=label3)
        label4 = tk.Label(self.root, text=" Jméno minulého snímku, kdyby jste v něm udělali chybu tak zapsat stranou : %s " % (self.last_filename))
        self.canvas.create_window(450,1000,window=label4)
        label5 = tk.Label(self.root, text=" Jméno snímku : %s " % (annotImg))
        self.canvas.create_window(450,1020,window=label5)
        #############
        ##FORM ENTRY#
        #############
        self.emo = tk.IntVar()
        self.emo.set(-1)
        self.val = tk.DoubleVar()
        self.val.set(-1)
        self.aro = tk.DoubleVar()
        self.aro.set(-1)
        valscale = tk.Scale(self.root, from_=-1., to=1., orient=tk.HORIZONTAL, variable=self.val, tickinterval=0.01,
                        digits = 3, resolution = 0.01, showvalue=0, command=self.move_dot)
        self.canvas.create_window(505,915,window=valscale, width=800)
        aroscale = tk.Scale(self.root, from_=1., to=-1., orient=tk.VERTICAL, variable=self.aro, tickinterval=0.01,
                        digits = 3, resolution = 0.01, showvalue=0, command=self.move_dot)
        self.canvas.create_window(975,457,window=aroscale, height=800)
        self.oval = self.canvas.create_oval(self.__x-10,self.__y-10,self.__x+10,self.__y+10,outline="#ff00ff",fill="#00ff00")
        self.move_dot("init")

        neutral = tk.Radiobutton(self.root,text="Neutrální", variable = self.emo, value=0)
        self.canvas.create_window(1040,570,window=neutral)

        happy = tk.Radiobutton(self.root,text="Veselý", variable = self.emo, value=1)
        self.canvas.create_window(1115,570,window=happy)

        angry = tk.Radiobutton(self.root,text="Rozzlobený", variable = self.emo, value=2)
        self.canvas.create_window(1190,570,window=angry)

        sad = tk.Radiobutton(self.root,text="Smutný", variable = self.emo, value=3)
        self.canvas.create_window(1265,570,window=sad)

        fear = tk.Radiobutton(self.root,text="Vystrašený", variable = self.emo, value=4)
        self.canvas.create_window(1340,570,window=fear)

        surprise = tk.Radiobutton(self.root,text="Překvapený", variable = self.emo, value=5)
        self.canvas.create_window(1430,570,window=surprise)

        disgust = tk.Radiobutton(self.root,text="Znechucený", variable = self.emo, value=6)
        self.canvas.create_window(1520,570,window=disgust)

        contempt = tk.Radiobutton(self.root,text="Pohrdavý", variable = self.emo, value=7)
        self.canvas.create_window(1605,570,window=contempt)


        submit = tk.Button(self.root, text = 'Pečlivě anotováno, ukaž další!', command=self.root.destroy)
        self.canvas.create_window(1500,800,window=submit)


        #############
        ## VAR IMG ##
        #############

        ToAnnotateImg = ImageTk.PhotoImage(Image.open("custom_nir/"+annotImg).resize((512,512),resample=2), master = self.canvas)  
        self.canvas.create_image(1100, 25, anchor=tk.NW, image=ToAnnotateImg)

        self.root.mainloop() 
        self.last_filename = annotImg
        return(self.emo.get(),self.val.get(),self.aro.get())
    
class Annotator:
    def __init__(self):
        self.GUI = GUI()
        self.df = pd.read_csv("custom_nir-filenames.csv")
    def annotate(self):
        emo = []
        val = []
        aro = []
        for i in range(0,len(self.df)):
            if i<50:
                print("skipping",i)
                continue
            print(i)           
            (tmp_emo,tmp_val,tmp_aro) = self.GUI.annotate_image(self.df.filename[i],
                    self.df.gender[i], self.df.adult[i], self.df.race[i], i+1,len(self.df) )
            emo.append(tmp_emo)
            val.append(tmp_val)
            aro.append(tmp_aro)
            
            # Add the new annotations to the DataFrame
            self.df.loc[i, "expression"] = tmp_emo
            self.df.loc[i, "valence"] = tmp_val
            self.df.loc[i, "arousal"] = tmp_aro
            # Save the DataFrame to a CSV file
            self.df.to_csv('custom_nir-annotations.csv', index=False)
            
        self.df["expression"]=emo
        self.df["valence"]=val
        self.df["arousal"]=aro
        self.df.to_csv(str("annotations"+datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+".csv"))
app = Annotator()
app.annotate()
