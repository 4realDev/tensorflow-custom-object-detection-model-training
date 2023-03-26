from PIL import Image
import cv2
import os
import customtkinter as ctk
from PIL import Image

import asyncio
import nest_asyncio
from miro_sticky_notes_sync import start_sticky_note_scanner

from miro_tfod_functions import \
    get_detections_from_img, \
    get_image_with_overlayed_labeled_bounding_boxes, \
    load_latest_checkpoint_of_custom_object_detection_model

nest_asyncio.apply()


# class Header(ctk.CTkFrame):
#     def __init__(self, master):
#         super().__init__(master)

#         self.width = 368
#         self.height = 24
#         self.fg_color = "#000000"
#         self.corner_radius = 0
#         # self.place(relx=0.5, rely=0.5, anchor="w")
#         self.pack(padx=0, pady=0, fill="both", expand=True)

class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global cap_width
        global cap_height
        global image_label
        global my_frame
        global preview_switcher

        def on_closing():
            preview_switcher.toggle()
            self.destroy()

        self.geometry(f"{cap_width}x{cap_height}")
        # self.resizable(False, False)
        self.title("Sticky Note Scanner Preview")

        self.protocol("WM_DELETE_WINDOW", on_closing)

        # self.label = ctk.CTkLabel(
        #     self,
        #     anchor="w",
        #     font=ctk.CTkFont(
        #         family="Heebo-Bold",
        #         size=20,
        #         weight="bold"),
        #     text="Sticky Note Scanner Preview",
        #     text_color="#050038",
        #     justify="left"
        # )
        # self.label.pack(padx=20, pady=20)

        my_frame = ctk.CTkFrame(
            master=self,
            width=cap_width,
            height=cap_height,
            fg_color="#000000",
            corner_radius=0,
        )

        my_frame.pack(padx=0, pady=0)

        image_label = ctk.CTkLabel(
            master=my_frame,
            width=cap_width,
            height=cap_height,
            text="",
            anchor="w",
            fg_color="#000000"
        )

        image_label.pack(padx=0, pady=0)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        global preview_switcher

        # Modes: system (default), light, dark
        ctk.set_appearance_mode("Light")
        # Themes: blue (default), dark-blue, green
        # ctk.set_default_color_theme("blue")

        self.title("Cando Toolbox / Sticky Note Scanner")
        self.geometry("368x731")
        # self.minsize(400, 300)
        # self.maxsize(1000, 800)
        self.resizable(True, True)
        self.configure(fg_color="#F3F3F3")

        self.my_frame = ctk.CTkFrame(
            master=self,
            width=368,
            height=24,
            fg_color="#000000",
            corner_radius=0,
        )
        self.my_frame.pack(padx=0, pady=0, fill="both",
                           expand=False, anchor="w")

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "gui_images")

        self.logo_image = ctk.CTkImage(
            light_image=Image.open(os.path.join(image_path, "Logo.png")),
            dark_image=Image.open(os.path.join(image_path, "Logo.png")),
            size=(184, 34)
        )
        # self.logo_image.place(relx=0.5, rely=0.5, anchor="center")
        # self.logo_image.pack(padx=20, pady=20)

        self.image_label = ctk.CTkLabel(
            master=self.my_frame,
            image=self.logo_image,
            width=184,
            height=34,
            text="",
            anchor="w",
            fg_color="#000000"
            # font=("Heebo-Medium", 14),
            # corner_radius=8,
            # text_color="#434343",
            # padx=150,
            # pady=150,
            # justify="left"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="w")
        self.image_label.pack(padx=24, pady=8, side="left")

        # ___________________________________________________________________________________________

        self.my_frame_2 = ctk.CTkFrame(
            master=self,
            # width=368,
            # height=24,
            # corner_radius=8,
            fg_color="#FFFFFF",
        )
        self.my_frame_2.pack(padx=24, pady=24, fill="both",
                             expand=True, anchor="w")

        # create 2x2 grid system
        self.my_frame_2.grid_rowconfigure(13, weight=1)
        self.my_frame_2.grid_columnconfigure((0, 1), weight=1)

        self.label = ctk.CTkLabel(
            master=self.my_frame_2,
            anchor="w",
            # width=320,
            # height=36,
            font=ctk.CTkFont(family="Heebo-Bold", size=20, weight="bold"),
            # corner_radius=8,
            text="Sticky Note Scanner",
            # fg_color=("#000000"),
            text_color="#050038",
            # padx=24,
            # pady=24,
            justify="left"
        )
        self.label.grid(row=1, column=0, columnspan=2,
                        padx=16, pady=(16, 24), sticky="nsew")

        self.entry_label = ctk.CTkLabel(
            master=self.my_frame_2,
            anchor="w",
            # width=320,
            # height=36,
            font=("Heebo-Medium", 14),
            # corner_radius=8,
            text="Miro Board Name:",
            # fg_color=("#000000"),
            text_color="#434343",
            # padx=24,
            # pady=24,
            justify="left"
        )
        self.entry_label.grid(row=2, column=0, columnspan=2,
                              padx=16, pady=0, sticky="nsew")

        # TODO: If useTimestamp fill field automatically with current timestamp and block entry
        # TODO: For this make the timestamp function as utility
        self.miro_board_name_entry = ctk.CTkEntry(
            master=self.my_frame_2,
            # textvariable=self.miro_board_name,
            placeholder_text="Name",
            placeholder_text_color="#434343",
            # anchor="center",
            width=320,
            height=36,
            font=("Heebo-Medium", 14),
            state="normal",
            border_color="#000000",
            border_width=1,
            # border_spacing=10,
            corner_radius=6,
            fg_color="#FFFFFF",
            text_color="#000000",
        )
        self.miro_board_name_entry.grid(row=3, column=0, columnspan=2,
                                        padx=16, pady=(4, 4), sticky="nsew")
        # self.entry.place(relx=0.5, rely=0.5, anchor="center")
        # self.entry.pack(padx=20, pady=20)

        self.timestamp_name_checkbox = ctk.CTkCheckBox(
            master=self.my_frame_2,
            # anchor="center",
            # width=320,
            checkbox_width=20,
            checkbox_height=20,
            # font=ctk.CTkFont(family="Heebo-Medium", size=12, weight="normal"),
            state="normal",
            border_color="#000000",
            border_width=2,
            # border_spacing=10,
            # corner_radius=6,
            text="Use timestamp as name",
            fg_color="#000000",
            text_color="#434343",
            # hover_color="pink",
            # variable=check_var,
            onvalue=1,
            offvalue=0,
            # value=1
            command=self.toggle_timestamp_checkbox
        )

        self.timestamp_name_checkbox.grid(row=4, column=0, columnspan=2,
                                          padx=16, pady=(0, 24), sticky="nw")

        self.create_new_miro_board_label = ctk.CTkLabel(
            master=self.my_frame_2,
            anchor="w",
            # width=320,
            # height=36,
            font=("Heebo-Medium", 14),
            # corner_radius=8,
            text="Miro Board Settings:",
            # fg_color=("#000000"),
            text_color="#434343",
            # padx=24,
            # pady=24,
            justify="left"
        )
        self.create_new_miro_board_label.grid(row=5, column=0, columnspan=2,
                                              padx=16, pady=0, sticky="nsew")
        # self.label.place(relx=0.5, rely=0.5, anchor="w")
        # self.label.pack(padx=20, pady=20)

        self.create_new_miro_board_radiobutton_var = ctk.IntVar(value=1)

        self.create_new_miro_board_radiobutton_1 = ctk.CTkRadioButton(
            master=self.my_frame_2,
            radiobutton_height=20,
            radiobutton_width=20,
            border_width_unchecked=2,
            border_width_checked=6,
            fg_color="#000000",
            border_color="#000000",
            text="Create new one",
            variable=self.create_new_miro_board_radiobutton_var,
            value=1
        )
        self.create_new_miro_board_radiobutton_1.grid(row=6, column=0, columnspan=2,
                                                      padx=16, pady=(4, 0), sticky="nw")
        # self.radiobutton_1.pack(pady=10, padx=10)

        self.create_new_miro_board_radiobutton_2 = ctk.CTkRadioButton(
            master=self.my_frame_2,
            radiobutton_height=20,
            radiobutton_width=20,
            border_width_unchecked=2,
            border_width_checked=6,
            fg_color="#000000",
            border_color="#000000",
            text="Work with existing",
            variable=self.create_new_miro_board_radiobutton_var,
            value=0
        )
        # self.radiobutton_2.pack(pady=10, padx=10)
        self.create_new_miro_board_radiobutton_2.grid(row=7, column=0, columnspan=2,
                                                      padx=16, pady=(4, 24), sticky="nw")

        self.create_new_frame_label = ctk.CTkLabel(
            master=self.my_frame_2,
            anchor="w",
            # width=320,
            # height=36,
            font=("Heebo-Medium", 14),
            # corner_radius=8,
            text="Frame Creation Settings:",
            # fg_color=("#000000"),
            text_color="#434343",
            # padx=24,
            # pady=24,
            justify="left"
        )
        self.create_new_frame_label.grid(row=8, column=0, columnspan=2,
                                         padx=16, pady=0, sticky="nsew")
        # self.label.place(relx=0.5, rely=0.5, anchor="w")
        # self.label.pack(padx=20, pady=20)

        self.create_new_frame_radiobutton_var = ctk.IntVar(value=1)

        self.create_new_frame_radiobutton_1 = ctk.CTkRadioButton(
            master=self.my_frame_2,
            radiobutton_height=20,
            radiobutton_width=20,
            text="Create new one",
            border_width_unchecked=2,
            border_width_checked=6,
            fg_color="#000000",
            border_color="#000000",
            variable=self.create_new_frame_radiobutton_var,
            value=1
        )
        self.create_new_frame_radiobutton_1.grid(row=9, column=0, columnspan=2,
                                                 padx=16, pady=(4, 0), sticky="nw")
        # self.radiobutton_1.pack(pady=10, padx=10)

        self.create_new_frame_radiobutton_2 = ctk.CTkRadioButton(
            master=self.my_frame_2,
            radiobutton_height=20,
            radiobutton_width=20,
            text="Overwrite existing",
            border_width_unchecked=2,
            border_width_checked=6,
            fg_color="#000000",
            border_color="#000000",
            variable=self.create_new_frame_radiobutton_var,
            value=0
        )
        # self.radiobutton_2.pack(pady=10, padx=10)
        self.create_new_frame_radiobutton_2.grid(row=10, column=0, columnspan=2,
                                                 padx=16, pady=(4, 24), sticky="nw")

        self.open_preview_label = ctk.CTkLabel(
            master=self.my_frame_2,
            anchor="w",
            # width=320,
            # height=36,
            font=("Heebo-Medium", 14),
            # corner_radius=8,
            text="Open Sticky Note Scanner Preview:",
            # fg_color=("#000000"),
            text_color="#434343",
            # padx=24,
            # pady=24,
            justify="left"
        )
        self.open_preview_label.grid(row=11, column=0, columnspan=2,
                                     padx=16, pady=0, sticky="nw")

        preview_switcher = ctk.CTkSwitch(
            master=self.my_frame_2,
            # anchor="center",
            # width=320,
            # checkbox_width=20,
            # checkbox_height=20,
            # font=ctk.CTkFont(family="Heebo-Medium", size=14, weight="normal"),
            state="normal",
            # border_color="blue",
            # border_width=2,
            # border_spacing=10,
            # corner_radius=6,
            text="Open Preview",
            fg_color="#000000",
            text_color="#434343",
            # hover_color="pink",
            # variable=check_var,
            # onvalue=0,
            # offvalue=1,
            command=self.open_toplevel
        )

        preview_switcher.grid(row=12, column=0, columnspan=2,
                              padx=16, pady=(0, 16), sticky="nw")

        self.button = ctk.CTkButton(
            master=self.my_frame_2,
            # anchor="center",
            # width=320,
            height=36,
            font=ctk.CTkFont(family="Heebo-Medium", size=14, weight="normal"),
            state="normal",
            # border_color="blue",
            # border_width=0,
            # border_spacing=10,
            corner_radius=6,
            text="Start scanning",
            fg_color="#000000",
            text_color="#FFFFFF",
            hover_color="pink",
            command=self.button_callback
        )
        self.button.grid(row=13, column=0, columnspan=2,
                         padx=16, pady=(0, 16), sticky="sew")

        # check_var = tkinter.StringVar("on")

        self.toplevel_window = None

    def toggle_timestamp_checkbox(self):
        print("toggle")

    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            # create window if its None or destroyed
            self.toplevel_window = ToplevelWindow(self)
        else:
            self.toplevel_window.focus()  # if window exists focus it

        # global my_frame
        global show_preview
        global my_frame

        show_preview = not show_preview

        if show_preview == True:
            # my_frame.pack(padx=0, pady=0, fill="both", expand=True, side="left")
            print("Video Streaming Started.")
            video_stream()
        else:
            # my_frame.pack_forget()
            self.toplevel_window.destroy()

    def button_callback(self):
        print("\n----- STARTING STICKY NOTE SCANNER MIRO SYNC -----\n")
        frame = video_stream()

        if self.toplevel_window is not None or self.toplevel_window.winfo_exists():
            self.toplevel_window.destroy()
            preview_switcher.toggle()

        asyncio.get_event_loop().run_until_complete(start_sticky_note_scanner(
            frame,
            self.miro_board_name_entry.get(),
            self.create_new_frame_radiobutton_var.get(),
            self.create_new_miro_board_radiobutton_var.get()
        ))
        print("\n----- COMPLETED STICKY NOTE SCANNER MIRO SYNC -----\n")


def video_stream():
    global image_label
    global show_preview

    _, frame = cap.read()
    frame_detections = get_detections_from_img(frame)
    frame_detections_np_with_detections = get_image_with_overlayed_labeled_bounding_boxes(
        frame,
        frame_detections,
    )

    repeat_function = None

    if image_label == None:
        return frame_detections_np_with_detections

    if show_preview == True:
        cv2image = cv2.cvtColor(
            frame_detections_np_with_detections, cv2.COLOR_BGR2RGBA)
        img = ctk.CTkImage(
            light_image=Image.fromarray(cv2image),
            dark_image=Image.fromarray(cv2image),
            size=(cap_width, cap_height)
        )
        image_label.configure(image=img)
        repeat_function = image_label.after(20, video_stream)
        return frame_detections_np_with_detections

    else:
        print("Video Streaming Stopped.")
        if repeat_function != None:
            image_label.after_cancel(repeat_function)
        return frame_detections_np_with_detections


my_frame = None
image_label = None
preview_switcher = None
cap = None
show_preview = False
cap_height = 0
cap_width = 0


async def main():

    load_latest_checkpoint_of_custom_object_detection_model()

    # global my_frame
    # global image_label
    global cap
    global cap_height
    global cap_width

    cap = cv2.VideoCapture(0)
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    app = App()

    app.mainloop()

    # async with aiohttp.ClientSession() as session:

    # await asyncio.create_task(scan_for_object_in_video())

    # ValueError: 'images' must have either 3 or 4 dimensions. -> could be related to wrong source of VideoCapture!

    # scan_condition = cap.isOpened()
    # video_stream()

    # while scan_condition:
    # await asyncio.sleep(1)
    # ret, frame = cap.read()
    # frame_detections = get_detections_from_img(frame)
    # frame_detections_np_with_detections = get_image_with_overlayed_labeled_bounding_boxes(
    #     frame,
    #     frame_detections,
    # )

    # cv2.imshow('object detection',  cv2.resize(
    #     frame_detections_np_with_detections, (800, 600)))

    # if keyboard.is_pressed("c"):
    #     print("Starting the creation of the miro backup.")
    #     await asyncio.create_task(run_miro_sync_process(frame, session))

    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     break


if __name__ == "__main__":
    asyncio.run(main())
