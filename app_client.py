import argparse
import json
import os
import io
import base64
import gradio as gr
from PIL import Image
import requests

import settings

TITLE = "# Nhóm 4 - Hệ thống và mạng máy tính"
DESCRIPTION = """
# Link data: [Oxbuild dataset](https://drive.google.com/drive/folders/1kihRTCj5CSO9oakkutR-fHV5F2YOKSet?usp=sharing)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


class SimilaritySearcher:
    def __init__(self, server_host, image_folder):
        self.predict_endpoint = server_host + '/predict'
        self.image_dir = image_folder
    def search(self, image, nr_retr):
        image_ids = []
        scores = []
        # results = [{'image': 'oxford_000898.jpg', 'score': 0.7307232022285461}, {'image': 'new_000735.jpg', 'score': 0.7292542457580566}, {'image': 'bodleian_000418.jpg', 'score': 0.7168529033660889}, {'image': 'oxford_002724.jpg', 'score': 0.7153230905532837}, {'image': 'new_000236.jpg', 'score': 0.7144255638122559}]
        results = []
        try:
            image = self.gr_image_to_bytes(image)
            response = requests.post(
                self.predict_endpoint,
                files={settings.FILE_KEY: image},
                data={settings.NR_RETR_KEY: nr_retr},
                timeout=settings.REQ_TIME_OUT,
            )
        
            if response.status_code == 200:
                response_rs =  response.json()
                results = response_rs["matched_files"]
                for item in results:
                    image_ids.append(item[settings.IMAGE_KEY])
                    scores.append(item[settings.SCORE_KEY])
                return self.get_image_by_ids(image_ids, scores)
            else:
                print("Request failed with status code:", response.status_code)
                return None
        except requests.Timeout:
            print("Request timed out after", settings.REQ_TIME_OUT, "seconds")
            return None
        except requests.RequestException as e:
            print("Request failed:", e)
            return None
                
    
    def get_image_by_ids(self, image_ids, scores):
        image_urls = []
        caps = []
        for (image_id, score) in zip(image_ids, scores):
            image_url = os.path.join(self.image_dir, image_id)
            cap = f'{image_id}:{score:.3f}'
            image_urls.append(image_url)
            caps.append(cap)
        return list(zip(image_urls, caps))

    @staticmethod
    def gr_image_to_bytes(image):
        # img = Image.fromarray(image.astype('uint8'), 'RGB')
        # img_bytes = io.BytesIO()
        # image.save(img_bytes, format='JPEG')
        # return img_bytes.getvalue()
        image_pil = image
        image_buffer = io.BytesIO()
        image_pil.save(image_buffer, format='JPEG')
        image_buffer.seek(0)
        return image_buffer.getvalue()


def main():
    args = parse_args()
    searcher = SimilaritySearcher(server_host=settings.SERVER_HOST, image_folder=settings.DB_IMAGE_FOLDER)
    print(searcher.predict_endpoint)

    with gr.Blocks() as demo:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            input = gr.Image(type="pil", label="Input")

            with gr.Column():
                # with gr.Row():
                #     api_username = gr.Textbox(label="API Username")
                #     api_key = gr.Textbox(label="API Key")
                # selected_ratings = gr.CheckboxGroup(
                #     choices=["General", "Sensitive", "Questionable", "Explicit"],
                #     value=["General", "Sensitive"],
                #     label="Ratings",
                # )
                with gr.Row():
                    n_neighbours = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1, label="Return images"
                    )
                find_btn = gr.Button("Find similar images")
        similar_images = gr.Gallery(label="Similar images")

        similar_images.style(grid=5)

        
        find_btn.click(
            fn=searcher.search,
            inputs=[
                input,
                n_neighbours,
            ],
            outputs=[similar_images],
        )

    # demo.queue()
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()