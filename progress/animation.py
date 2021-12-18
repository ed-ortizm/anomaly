import imageio
import glob

from sdss.superclasses import FileDirectory

check = FileDirectory()

frames_per_second = 250

for filt in ["nofilter"]:  # , "nofilter"]:

    for relative in ["relative", "norelative"]:

        for data_type in ["normal", "anomaly"]:  # , "middle"]:

            images = glob.glob(f"images/{filt}/{relative}/{data_type}/*")
            if data_type == "anomaly":

                images.sort(reverse=True)  # to see outliers firs
            elif data_type == "normal":

                images.sort()

            save_to = f"images/animation/{frames_per_second}/{data_type}"
            check.check_directory(save_to, exit=False)

            file_name = f"{filt}_{relative}_{data_type}_{frames_per_second}"

            with imageio.get_writer(
                f"{save_to}/{file_name}.gif", mode="I", fps=frames_per_second
            ) as w:

                for idx, image in enumerate(images):

                    print(f"{idx}--> {filt}_{relative}_{data_type}", end="\r")

                    data = imageio.imread(image)
                    w.append_data(data)
