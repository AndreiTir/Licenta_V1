# made by Tir

from tkinter import *
import os
import cv2
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from tkinter import messagebox
sys.path.append('sort')
from sort.sort import *
if os.name == 'posix':
    import tflite_runtime.interpreter as tflite
elif os.name == 'nt':
    import tensorflow as tf
    assert tf.__version__.startswith('2')
    tf.get_logger().setLevel('ERROR')


if os.name == 'nt':
    faces_tflite = 'efficientdet-lite-faces.tflite'
    faces_labels = 'labels.txt'
    persons_tflite = 'efficientdet-lite-persons.tflite'
    persons_labels = 'labels-persons.txt'
elif os.name == 'posix':
    # nu merge cu edgetpu
    faces_tflite = 'efficientdet-lite-faces_edgetpu.tflite'
    faces_labels = 'labels.txt'
    persons_tflite = 'efficientdet-lite-persons_edgetpu.tflite'
    persons_labels = 'labels-persons.txt'

mot_tracker = Sort()

tflite_model_name = persons_tflite
tflite_labels_name = persons_labels
main_x = 320.5
main_y = 240.5

if os.name == 'posix':
    from periphery import PWM
    pwm_x = PWM(0, 0)  # (Pin 32)
    pwm_y = PWM(1, 0)  # (Pin 33)
    pwm_x.frequency = 50
    pwm_y.frequency = 50


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        #cap.release()


def draw_box(img, objs, scale_factor, labels, track_bbs_ids):
    """
    Deseneaza chenarele in jurul obiectelor si afiseaza probabilitatea si id-ul.

    Parameters
    -------
    img: numpy.ndarray
        imaginea pe care trebuie desenat
    objs: list
        obiectele prezente in cadru
    scale_factor: float
        raportul dintre dimensiunile intitiale si cele noi
    labels: dict
        etichetele utilizate
    track_bbs_ids: list
        lista de id-uri si pozitia corespunzatoare
    """
    dict = {}
    id_uri = [a[4] for a in track_bbs_ids]
    for j in track_bbs_ids:
        box_uri = [o.bbox for o in objs]
        for k in box_uri:
            if j[0] == k.xmin and j[1] == k.ymin and j[2] == k.xmax and j[3] == k.ymax:
                dict[box_uri.index(k)] = j[4]
    color = (0, 255, 0)
    for obj in objs:
        bbox = obj.bbox
        start_point = (int(bbox.xmin * scale_factor), int(bbox.ymin * scale_factor))
        end_point = (int(bbox.xmax * scale_factor), int(bbox.ymax * scale_factor))
        cx = (start_point[0]+end_point[0])//2
        cy = (start_point[1]+end_point[1])//2
        if len(objs) > 0:
            # selectare pozi
            pozi = None
            for i in range(len(objs)):
                if ((bbox.xmin + bbox.xmax) / 2 == (objs[i].bbox.xmin + objs[i].bbox.xmax) / 2) and \
                        ((bbox.ymin + bbox.ymax) / 2 == (objs[i].bbox.ymin + objs[i].bbox.ymax) / 2):
                    pozi = i
            #print(id_uri, dict)
            #pozi2 = id_uri.index(dict[objs.index(obj)])
            #print(pozi, pozi2)
            try:
                cv2.putText(
                    img,
                    '%s %d %.2f' % (labels.get(obj.id, obj.id), track_bbs_ids[pozi][4], obj.score),
                    (int(bbox.xmin * scale_factor - 5), int(bbox.ymin * scale_factor - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.rectangle(img, start_point, end_point, color, 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            except IndexError:
                pass


class ServoPosition:
    def __init__(self):
        # 0.015 = 0deg, 0.114 = +180deg, cu pas de 0.002
        self.curr_x = 0.   # 0deg
        self.curr_y = 0.   # 0deg


servo = ServoPosition()


def get_target(track_bbs_ids, id=0):
    global main_x, main_y
    center_x = 320.5
    center_y = 240.5
    delta_x = 0
    delta_y = 0
    #print("Really", end=' ')
    try:
        main_x = (track_bbs_ids[id][0] + track_bbs_ids[id][2]) / 2
        main_y = (track_bbs_ids[id][1] + track_bbs_ids[id][3]) / 2
        delta_x = round(center_x - main_x, 2)
        delta_y = round(center_y - main_y, 2)
        print(delta_x, delta_y)
        return delta_x, delta_y
    except IndexError:
        pass


    def change_pos(val):
        pass

    def tracking_servo():
        global main_x, main_y, integral_x, integral_y, differential_x, differential_y
        global prev_x, prev_y, track_bbs_ids, delta_x, delta_y, px, ix, dx, py, iy, dy
        valx = px * delta_x + dx * differential_x + ix * integral_x
        valy = py * delta_y + dy * differential_y + iy * integral_y

        valx = round(valx, 2)  # round off to 2 decimal points.
        valy = round(valy, 2)
        """
            if abs(delta_x) < 20:
                ser.setdcx(0)
            else:
                if abs(valx) > 0.5:
                    sign = valx / abs(valx)
                    valx = 0.5 * sign
                ser.setposx(valx)

            if abs(delta_y) < 20:
                ser.setdcy(0)
            else:
                if abs(valy) > 0.5:
                    sign = valy / abs(valy)
                    valy = 0.5 * sign
                ser.setposy(valy)"""


class App:
    def __init__(self, parent):
        self.parent = parent
        self.label = Label(self.parent, text="In ce context va fi utilizat algoritmul?", font=('DejavuSans', 12))
        self.label.pack(pady=20)
        self.button1 = Button(self.parent, text='Interior', font=('DejavuSans', 12), command=self.new_win_1)
        self.button1.pack(pady=20)
        self.button2 = Button(self.parent, text='Exterior', font=('DejavuSans', 12), command=self.new_win_2)
        self.button2.pack(pady=20)

    def new_win_1(self):
        global tflite_labels_name, tflite_model_name
        cap = cv2.VideoCapture(0)

        def on_closing_a():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                cap.release()
                new_win.destroy()

        def change_model():
            global tflite_labels_name, tflite_model_name, labels, interpreter
            if clicked.get() == 'Person detection':
                tflite_labels_name = persons_labels
                tflite_model_name = persons_tflite
            elif clicked.get() == 'Face detection':
                tflite_labels_name = faces_labels
                tflite_model_name = faces_tflite
            labels = read_label_file(tflite_labels_name)
            if os.name == 'posix':
                interpreter = tflite.Interpreter(tflite_model_name,
                                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            elif os.name == 'nt':
                interpreter = tf.lite.Interpreter(tflite_model_name)
            interpreter.allocate_tensors()

        def callback(selection):
            # print(clicked.get() == 'Face detection')
            change_model()

        new_win = Toplevel(self.parent)
        new_win.title("Interior")
        new_win.geometry("900x720")
        new_win.configure(bg="black")
        #f1 = LabelFrame(new_win, bg="black")
        #f1.pack(pady=50)
        m1 = Label(new_win, bg="black")
        m1.pack(pady=50)
        options = [
            "Person detection",
            "Face detection"
        ]
        clicked = StringVar(new_win)
        clicked.set(options[0])
        drop = OptionMenu(new_win, clicked, *options, command=callback)
        drop.pack(pady=30)
        while True:
            labels = read_label_file(tflite_labels_name)
            if os.name == 'posix':
                interpreter = tflite.Interpreter(tflite_model_name,
                                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            elif os.name == 'nt':
                interpreter = tf.lite.Interpreter(tflite_model_name)
            interpreter.allocate_tensors()

            display_width = 640
            try:
                img = cap.read()[1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                _, scale = common.set_resized_input(
                    interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
                interpreter.invoke()
                objs = detect.get_objects(interpreter, score_threshold=0.55, image_scale=scale)

                boxes = []
                centers = []
                for obj in objs:
                    boxes.append(np.array([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax, obj.score]))
                    centers.append([(obj.bbox.xmin + obj.bbox.xmax)/2, (obj.bbox.ymin + obj.bbox.ymax)/2])
                if objs:
                    track_bbs_ids = mot_tracker.update(np.array(boxes))
                    get_target(track_bbs_ids)

                scale_factor = display_width / img.width
                height_ratio = img.height / img.width
                img.resize((display_width, int(display_width * height_ratio)))

                img = np.asarray(img)
                if objs:
                    draw_box(img, objs, scale_factor, labels, track_bbs_ids)

                img = ImageTk.PhotoImage(Image.fromarray(img))

                try:
                    m1['image'] = img
                except:
                    break

                if objs:
                    #tracking_servo()
                    pass
                new_win.protocol("WM_DELETE_WINDOW", on_closing_a)
                new_win.update()
            except:
                break

    def new_win_2(self):
        global tflite_labels_name, tflite_model_name, persons_tflite, persons_labels
        tracking = 0
        cap = cv2.VideoCapture(0)
        tflite_labels_name = persons_labels
        tflite_model_name = persons_tflite
        labels = read_label_file(tflite_labels_name)
        if os.name == 'posix':
            interpreter = tflite.Interpreter(tflite_model_name,
                                             experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        elif os.name == 'nt':
            interpreter = tf.lite.Interpreter(tflite_model_name)
        interpreter.allocate_tensors()

        def on_closing_a():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                cap.release()
                new_win.destroy()

        def callback(selection):
            #print(clicked.get())
            pass

        def change_state():
            if check_var.get() == 1:
                drop.configure(state='normal')
                tracking = 1
                modificare_lista(track_bbs_ids)
            else:
                drop.configure(state='disabled')
                tracking = 0

        def update_option_menu():
            global options
            menu = drop["menu"]
            menu.delete(0, "end")
            for string in options:
                menu.add_command(label=string,
                                 command=lambda value=string: clicked.set(value))
            if len(options) == 1 or clicked.get() not in options:
                clicked.set(options[0])

        def modificare_lista(tracking):
            global options
            a = ['-']
            for j in tracking:
                a.append(str(int(j[4])))
            options = a
            print(options)
            update_option_menu()

        new_win = Toplevel(self.parent)
        new_win.title("Exterior")
        new_win.geometry("900x720")
        new_win.configure(bg="black")

        m1 = Label(new_win, bg="black")
        m1.pack(pady=25)

        check_var = IntVar()
        check_var.set(0)
        cb = Checkbutton(
            new_win,
            text="Activare/Dezactivare alegere manuala",
            variable=check_var,
            onvalue=1,
            offvalue=0,
            height=2,
            width=40,
            command=change_state
        )
        cb.pack(expand=True)

        options = ['-']
        clicked = StringVar()
        drop = OptionMenu(new_win, clicked, *options, command=callback)
        drop.configure(state='disabled')
        drop.pack(pady=30)
        display_width = 640

        while True:
            try:
                img = cap.read()[1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                _, scale = common.set_resized_input(
                    interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
                interpreter.invoke()
                objs = detect.get_objects(interpreter, score_threshold=0.55, image_scale=scale)

                boxes = []
                centers = []
                for obj in objs:
                    boxes.append(np.array([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax, obj.score]))
                    centers.append([(obj.bbox.xmin + obj.bbox.xmax) / 2, (obj.bbox.ymin + obj.bbox.ymax) / 2])
                if objs:
                    track_bbs_ids = mot_tracker.update(np.array(boxes))
                    if tracking == 1 and clicked.get() != '-':
                        id_uri = [int(tra[4]) for tra in track_bbs_ids]
                        id_urmarit = id_uri.index(int(clicked.get()))
                        get_target(track_bbs_ids, id_urmarit)
                    elif tracking == 0:
                        get_target(track_bbs_ids)

                scale_factor = display_width / img.width
                height_ratio = img.height / img.width
                img.resize((display_width, int(display_width * height_ratio)))

                img = np.asarray(img)
                if objs:
                    draw_box(img, objs, scale_factor, labels, track_bbs_ids)
                img = ImageTk.PhotoImage(Image.fromarray(img))
                modificare_lista(track_bbs_ids)
                try:
                    m1['image'] = img
                except:
                    pass
            except:
                break
            new_win.protocol("WM_DELETE_WINDOW", on_closing_a)
            new_win.update()


if __name__ == "__main__":
    #cap = cv2.VideoCapture(0)
    root = Tk()
    root.geometry('380x240')
    root.title("Alegerea contextului de functionare")
    App(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.bind('<Escape>', lambda e: root.destroy())
    root.mainloop()
    #cap.release()
