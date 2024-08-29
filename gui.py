from customtkinter import *
from tkinter import filedialog
import threading as th


from Model.Sources.youtube import YouTubeVideo

# creating the window
app = CTk()
app.title("Auto subtitle generator")
# window resolution
app.geometry("500x400")


clips_path = "./dataset/youtube/{}/clips"
shorts_path = "./dataset/youtube/{}/shorts"
youtube_path = "./dataset/youtube"


class Gui:
    def __init__(self, app) -> None:
        self.app = app
        self.url = ""
        self.data_downloaded = False
        self.clips_exported = False
        self.title = ""
        # Place Home
        self.Home()
        # this label is at the very bottom and gives feedback to the player that the program is still running
        self.dotsLabel = CTkLabel(
            master=self.app, text="", font=("Arial", 20), text_color="#FFCC70"
        )
        self.dotsLabel.place(relx=0.5, rely=0.8, anchor="center")

        # Creating and placing the label that tells the user if a video is being created
        self.processLabel = CTkLabel(
            master=self.app, text="", font=("Arial", 20), text_color="#FFCC70"
        )
        self.processLabel.place(relx=0.5, rely=0.7, anchor="center")

        self.videoLabel = CTkLabel(
            master=self.app, text="", font=("Arial", 20), text_color="#FFCC70"
        )
        self.videoLabel.place(relx=0.5, rely=0.5, anchor="center")

        self.dots = "."  # a string that will be used for the Dots() method
        self.count = 0
        self.processing = False

    def Home(self):
        # Label to enter a url
        self.url_label = CTkLabel(
            master=app,
            text="Auto-Transcriber",
            font=("Arial", 32),
            text_color="#FFCC70",
        )
        self.url_label.place(relx=0.5, rely=0.3, anchor="center")

        # Add an input field (CTkEntry)
        self.url_entry = CTkEntry(master=app, width=450, font=("Arial", 18))
        self.url_entry.place(relx=0.5, rely=0.4, anchor="center")

        # Set URL
        self.setButton = CTkButton(
            master=self.app,
            text="Confirm",
            command=self.set_url,
            fg_color="orange",
            text_color="black",
            width=80,
        )
        self.setButton.place(relx=0.84, rely=0.4, anchor="center")

    def ExportPage(self):
        if self.url != "":
            # Download button
            self.download_btn = CTkButton(
                master=self.app,
                text="Download",
                command=self.start_video_download,
                fg_color="orange",
                text_color="black",
            )
            self.download_btn.place(relx=0.3, rely=0.6, anchor="center")
            # Export Button
            self.export_btn = CTkButton(
                master=self.app,
                text="Export Clips",
                command=self.start_clips_export,
                fg_color="orange",
                text_color="black",
            )
            self.export_btn.place(relx=0.5, rely=0.6, anchor="center")
            # Browse Button
            self.browseButton = CTkButton(
                master=self.app,
                text="Browse Clips",
                command=self.browse_clips,
                fg_color="orange",
                text_color="black",
                width=10,
                font=("Arial", 12),
            )
            self.browseButton.place(relx=0.8, rely=0.05, anchor="center")
            # Browse Shorts
            self.browseShortsButton = CTkButton(
                master=self.app,
                text="Browse Shorts",
                command=self.browse_shorts,
                fg_color="orange",
                text_color="black",
                width=10,
                font=("Arial", 12),
            )
            self.browseShortsButton.place(relx=0.93, rely=0.05, anchor="center")

            # Create Shorts
            self.createShorts = CTkButton(
                master=self.app,
                text="Create Shorts",
                command=self.start_shorts_export,
                fg_color="orange",
                text_color="black",
            )

            self.createShorts.place(relx=0.7, rely=0.6, anchor="center")

    # ===============================
    # Set Url
    # ===============================
    def set_url(self):
        self.url = self.url_entry.get()
        self.yt = YouTubeVideo(url=self.url)
        self.videoLabel.configure(text=f"{self.yt.title}")
        self.videoLabel.update()
        self.ExportPage()

    # ===============================
    # Video Download
    # ===============================
    def start_video_download(self):
        processThread = th.Thread(target=self.download_video)
        processThread.start()
        dotsThread = th.Thread(target=self.Dots)
        self.processing = True
        dotsThread.start()

    def download_video(self):
        self.processLabel.configure(text="Downloading Video Information... ")
        self.processLabel.update()
        y = YouTubeVideo(url=self.url_entry.get(), log=True)
        self.title = y.title
        self.videoLabel.configure(text=f"{y.title}")
        self.videoLabel.update()
        y.download_youtube_video_info()
        self.processLabel.configure(text="Finished Downloading... ")
        self.processLabel.update()
        self.processing = False
        self.count = 0

    # ===============================
    # Clip Export
    # ===============================
    def start_clips_export(self):
        processThread = th.Thread(target=self.export_clips)
        processThread.start()
        dotsThread = th.Thread(target=self.Dots)
        self.processing = True
        dotsThread.start()

    def export_clips(self):

        y = YouTubeVideo(url=self.url_entry.get(), log=True)
        self.title = y.title
        self.videoLabel.configure(text=f"{y.title}")
        self.videoLabel.update()

        data_exists = y._check_if_data_downloaded()

        if data_exists:

            self.processLabel.configure(text="Exporting Clips... ")
            self.processLabel.update()
            y.export_all_clips()
            self.processLabel.configure(text="Finished Exporting Clips... ")
            self.processLabel.update()

        else:
            self.processLabel.configure(
                text="Local Data Not Found. Attempting to Download... "
            )
            self.processLabel.update()
            y.download_youtube_video_info()
            new_data_exists = y._check_if_data_downloaded()

            if new_data_exists:
                self.processLabel.configure(text="Exporting Clips... ")
                self.processLabel.update()
                y.export_all_clips()
                self.processLabel.configure(text="Finished Exporting Clips... ")
                self.processLabel.update()

            else:
                self.processLabel.configure(text="Failed to Download... ")
                self.processLabel.update()
        self.processing = False
        self.count = 0

    # ===============================
    # Clip Browsing
    # ===============================
    def browse_clips(self):
        if self.title == "":
            try:
                url = self.url_entry.get()
                y = YouTubeVideo(url)
                title = y.title
                path = clips_path.format(title)
            except Exception:
                path = youtube_path
        else:
            path = clips_path.format(self.title)

        file_path = filedialog.askopenfilename(
            initialdir=path,
            title="Select an MP4 file",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
        )

    def browse_shorts(self):
        if self.title == "":
            try:
                url = self.url_entry.get()
                y = YouTubeVideo(url)
                title = y.title
                path = shorts_path.format(title)
            except Exception:
                path = youtube_path
        else:
            path = shorts_path.format(self.title)

        file_path = filedialog.askopenfilename(
            initialdir=path,
            title="Select an MP4 file",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
        )

    # ===============================
    # Create Shorts
    # ===============================
    def start_shorts_export(self):
        processThread = th.Thread(target=self.export_shorts)
        processThread.start()
        dotsThread = th.Thread(target=self.Dots)
        self.processing = True
        dotsThread.start()

    def export_shorts(self):
        y = YouTubeVideo(url=self.url_entry.get(), log=True)
        self.title = y.title
        self.videoLabel.configure(text=f"{y.title}")
        self.videoLabel.update()

        data_exists = y._check_if_data_downloaded()

        if data_exists:

            self.processLabel.configure(text="Creating Shorts... ")
            self.processLabel.update()
            y.convert_to_short()
            self.processLabel.configure(text="Finished Creating Shorts... ")
            self.processLabel.update()

        else:
            self.processLabel.configure(
                text="Local Data Not Found. Attempting to Download... "
            )
            self.processLabel.update()
            y.download_youtube_video_info()
            new_data_exists = y._check_if_data_downloaded()

            if new_data_exists:
                self.processLabel.configure(text="Creating Shorts... ")
                self.processLabel.update()
                y.convert_to_short()
                self.processLabel.configure(text="Finished Creating Shorts... ")
                self.processLabel.update()

            else:
                self.processLabel.configure(text="Failed to Download... ")
                self.processLabel.update()
        self.processing = False
        self.count = 0

    def Dots(
        self,
    ):  # This method gives feedback to the player that the program is still running
        if self.processing:
            if self.count > 15:
                self.count = -1
                self.dots = "."
                self.dotsLabel.configure(text=self.dots)
                self.dotsLabel.after(100, self.Dots)
            else:
                try:
                    self.dots = self.dots + self.dots[self.count]
                except IndexError:
                    self.count = 0
                    self.dots = self.dots + self.dots[self.count]
                self.dotsLabel.configure(text=self.dots)
                self.count += 1
                self.dotsLabel.after(100, self.Dots)
        else:
            self.dots = ""
            self.dotsLabel.configure(text=self.dots)


# def Browse():
#     video_path = filedialog.askopenfilename()


# def StartVideoProcess():
#     pass


# def get_input():
#     input_value = entry.get()

#     y = YouTubeVideo(url=input_value)
#     y.download_youtube_video_info()


# label = CTkLabel(
#     master=app, text="Select model", font=("Arial", 20), text_color="#FFCC70"
# )
# label.place(relx=0.5, rely=0.3, anchor="center")

# model = CTkComboBox(master=app, values=["Whisper", "Model 1", "Model 2"])
# model.place(relx=0.5, rely=0.4, anchor="center")

# font = CTkComboBox(master=app, values=["Arial", "Times New Roman", "Courier"])
# font.place(relx=0.5, rely=0.5, anchor="center")

# processBtn = CTkButton(master=app, text="Process", command=StartVideoProcess)
# processBtn.place(relx=0.5, rely=0.7, anchor="center")

if __name__ == "__main__":

    app = CTk()

    app.title("Auto subtitle generator")
    # window resolution
    app.geometry("800x600")

    gui = Gui(app)

    app.mainloop()
