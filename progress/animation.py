import imageio
import glob

frames_per_second = 400

for filt in ["filter", "nofilter"]:

    for relative in ["relative", "norelative"]:

        for data_type in ["normal", "anomaly", "middle"]:

            images = glob.glob(f"images/{filt}/{relative}/{data_type}/*")
            images.sort()
            save_to = f"images/animation/{filt}_{relative}_{data_type}_{frames_per_second}"

            with imageio.get_writer(
                f'{save_to}.gif',
                mode='I',
                fps=frames_per_second
            ) as w:

                for idx, image in enumerate(images):
                
                    print(f"{idx}--> {filt}_{relative}_{data_type}", end="\r")

                    data = imageio.imread(image)
                    w.append_data(data)
