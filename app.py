import os
import json
import random
import spacy
from typing import List, Dict
import base64
from io import BytesIO
from datetime import datetime
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import numpy as np
from textblob import TextBlob
from moviepy import VideoFileClip

class CaptionHighlighter:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy English model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def highlight_important_words(self, caption: str) -> str:
        doc = self.nlp(caption)
        highlighted_parts = []
        last_end = 0
        for token in doc:
            highlighted_parts.append(caption[last_end:token.idx])
            if token.ent_type_:
                highlighted_parts.append(f'<mark style="color: magenta; background-color: transparent;">{token.text}</mark>')
            elif token.pos_ == "PROPN":
                highlighted_parts.append(f'<mark style="color: lightblue; background-color: transparent;">{token.text}</mark>')
            elif token.pos_ == "NOUN":
                highlighted_parts.append(f'<mark style="color: lightgreen; background-color: transparent;">{token.text}</mark>')
            elif token.pos_ == "VERB":
                highlighted_parts.append(f'<mark style="color: lightcoral; background-color: transparent;">{token.text}</mark>')
            else:
                highlighted_parts.append(token.text)
            last_end = token.idx + len(token.text)
        highlighted_parts.append(caption[last_end:])
        return ''.join(highlighted_parts)

class CaptionRatingSystem:
    BASE_RATING = 1500
    BASE_K = 32

    def calculate_dynamic_k(self, comparisons):
        return max(self.BASE_K / (1 + comparisons / 10), 8)

    def calculate_elo_rating(self, rating1, rating2, result, comparisons1, comparisons2):
        k_factor1 = self.calculate_dynamic_k(comparisons1)
        k_factor2 = self.calculate_dynamic_k(comparisons2)
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        new_rating1 = rating1 + k_factor1 * (result - expected1)
        new_rating2 = rating2 + k_factor2 * ((1 - result) - expected2)
        return new_rating1, new_rating2

class VideoCaptionApp:
    def __init__(self, videos_dir, captions_all_dir, captions_best_dir):
        self.videos_dir = videos_dir
        self.captions_all_dir = captions_all_dir
        self.captions_best_dir = captions_best_dir
        self.rating_system = CaptionRatingSystem()
        self.caption_highlighter = CaptionHighlighter()
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(captions_all_dir, exist_ok=True)
        os.makedirs(captions_best_dir, exist_ok=True)
        self.captions_db = self.load_existing_captions(captions_all_dir)
        self.video_list = self.get_video_list()

    def load_existing_captions(self, captions_dir):
        captions_db = {}
        for filename in os.listdir(captions_dir):
            if filename.endswith('.json'):
                video_id = os.path.splitext(filename)[0]
                try:
                    with open(os.path.join(captions_dir, filename), 'r') as f:
                        captions = json.load(f)
                    captions_db[video_id] = captions
                except json.JSONDecodeError:
                    print(f"Error reading caption file {filename}")
        return captions_db

    def get_video_list(self):
        return [f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    def get_random_video(self):
        return random.choice(self.video_list) if self.video_list else None

    def get_video_path(self, video_filename):
        return os.path.join(self.videos_dir, video_filename)

    def compare_captions(self, video_id, caption1, caption2, user_preference):
        captions = self.captions_db.get(video_id, [])
        idx1 = next((i for i, c in enumerate(captions) if c['text'] == caption1), -1)
        idx2 = next((i for i, c in enumerate(captions) if c['text'] == caption2), -1)
        if idx1 == -1 or idx2 == -1:
            return "Captions not found"
        rating1 = captions[idx1].get('elo_rating', self.rating_system.BASE_RATING)
        rating2 = captions[idx2].get('elo_rating', self.rating_system.BASE_RATING)
        comparisons1 = captions[idx1].get('comparisons', 0)
        comparisons2 = captions[idx2].get('comparisons', 0)
        result = 1 if user_preference == 1 else 0
        new_rating1, new_rating2 = self.rating_system.calculate_elo_rating(
            rating1, rating2, result, comparisons1, comparisons2
        )
        captions[idx1]['elo_rating'] = new_rating1
        captions[idx2]['elo_rating'] = new_rating2
        captions[idx1]['comparisons'] = comparisons1 + 1
        captions[idx2]['comparisons'] = comparisons2 + 1
        self.save_captions(video_id, captions)
        self.update_best_captions()
        return "Comparison recorded"

    def add_caption(self, video_id, new_caption):
        if video_id not in self.captions_db:
            self.captions_db[video_id] = []
        new_caption_entry = {
            'text': new_caption,
            'timestamp': datetime.now().isoformat(),
            'elo_rating': self.rating_system.BASE_RATING,
            'comparisons': 0,
            'id': len(self.captions_db[video_id])
        }
        self.captions_db[video_id].append(new_caption_entry)
        self.save_captions(video_id, self.captions_db[video_id])
        self.update_best_captions()
        return "Caption added"

    def get_top_captions(self, video_id, top_n=3):
        captions = self.captions_db.get(video_id, [])
        sorted_captions = sorted(captions, key=lambda x: x.get('elo_rating', 0), reverse=True)
        for caption in sorted_captions:
            caption['highlighted_text'] = self.caption_highlighter.highlight_important_words(caption['text'])
        return sorted_captions[:top_n]

    def save_captions(self, video_id, captions):
        filepath = os.path.join(self.captions_all_dir, f"{video_id}.json")
        with open(filepath, 'w') as f:
            json.dump(captions, f, indent=2)

    def update_best_captions(self):
        for video_id, captions in self.captions_db.items():
            if captions:
                sorted_captions = sorted(captions, key=lambda x: x.get('elo_rating', 0), reverse=True)
                best_caption = sorted_captions[0]
                best_caption_text = best_caption['text']
                filepath = os.path.join(self.captions_best_dir, f"{video_id}.txt")
                with open(filepath, 'w') as f:
                    f.write(best_caption_text)

    def create_gradio_interface(self):
        with gr.Blocks(css="""
        .gradio-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            width: 80%;
            max-width: 800px; /* Adjust as needed */
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        video {
            max-width: 100%;
            max-height: 80vh;
        }
        .captions-container {
            width: 80%;
            max-width: 800px; /* Match the video container width */
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }
        .highlighted-caption {
            line-height: 2;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }
        .highlighted-caption mark {
            padding: 2px;
            border-radius: 3px;
        """) as demo:
            video_id = gr.State(None)
            current_captions = gr.State([])
            new_caption_input = gr.State("")

            with gr.Tab("Caption Comparison"):
                video_display = gr.Video(autoplay=False)

                with gr.Row():
                    caption1 = gr.HTML(label="Caption 1", elem_classes=["highlighted-caption"])
                    caption2 = gr.HTML(label="Caption 2", elem_classes=["highlighted-caption"])
                with gr.Row():
                    compare_btn1 = gr.Button("Prefer Caption 1")
                    compare_btn2 = gr.Button("Prefer Caption 2")
                with gr.Row():
                    new_caption_input = gr.Textbox(label="Add New Caption")
                    submit_caption_btn = gr.Button("Submit Caption")
                random_video_btn = gr.Button("Get Random Video")

                def load_random_video():
                    video_filename = self.get_random_video()
                    if not video_filename:
                        return None, "No videos available", None, [], ""
                    video_path = self.get_video_path(video_filename)
                    video_id_str = os.path.splitext(video_filename)[0]
                    top_captions = self.get_top_captions(video_id_str)
                    caption_texts = [f"{c.get('highlighted_text', '')} (Elo: {c.get('elo_rating', 0)})" for c in top_captions]
                    while len(caption_texts) < 2:
                        caption_texts.append("")
                    return video_path, caption_texts[0], caption_texts[1], video_id_str, top_captions, ""

                random_video_btn.click(
                    fn=load_random_video,
                    outputs=[
                        video_display, caption1, caption2, video_id, current_captions, new_caption_input
                    ]
                )
                demo.load(
                    fn=load_random_video,
                    outputs=[
                        video_display, caption1, caption2, video_id, current_captions, new_caption_input
                    ]
                )

                def handle_comparison(video_id, captions, preference):
                    if not captions or len(captions) < 2:
                        return "Not enough captions to compare"
                    caption1_text = captions[0]['text'] if captions else ""
                    caption2_text = captions[1]['text'] if len(captions) > 1 else ""
                    result = self.compare_captions(video_id, caption1_text, caption2_text, preference)
                    return load_random_video()

                compare_btn1.click(
                    fn=lambda video_id, captions: handle_comparison(video_id, captions, 1),
                    inputs=[video_id, current_captions],
                    outputs=[
                        video_display, caption1, caption2, video_id, current_captions, new_caption_input
                    ]
                )
                compare_btn2.click(
                    fn=lambda video_id, captions: handle_comparison(video_id, captions, 2),
                    inputs=[video_id, current_captions],
                    outputs=[
                        video_display, caption1, caption2, video_id, current_captions, new_caption_input
                    ]
                )

                def submit_new_caption(video_id, new_caption):
                    if not video_id or not new_caption:
                        return "Invalid input"
                    self.add_caption(video_id, new_caption)
                    return load_random_video()

                submit_caption_btn.click(
                    fn=submit_new_caption,
                    inputs=[video_id, new_caption_input],
                    outputs=[
                        video_display, caption1, caption2, video_id, current_captions, new_caption_input
                    ]
                )

            with gr.Tab("Statistics"):
                refresh_stats_btn = gr.Button("Refresh Stats")
                stats_report = gr.HTML()

                def generate_thumbnail(video_path):
                    clip = VideoFileClip(video_path)
                    frame = clip.get_frame(0)  # Get the first frame
                    buffered = BytesIO()
                    plt.imsave(buffered, frame)
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    return img_str

                def generate_stats_report():
                    top_n = 3  # Number of top captions to display per video
                    html_report = "<h2>Top Captions per Video</h2>\n"
                    html_report += "<table>\n"
                    html_report += "    <thead>\n"
                    html_report += "        <tr>\n"
                    html_report += "            <th>Video Thumbnail</th>\n"
                    html_report += "            <th>Video ID</th>\n"
                    html_report += "            <th>Caption</th>\n"
                    html_report += "            <th>ELO Rating</th>\n"
                    html_report += "        </tr>\n"
                    html_report += "    </thead>\n"
                    html_report += "    <tbody>\n"

                    for video_id, captions in self.captions_db.items():
                        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")  # Adjust extension if needed
                        if os.path.exists(video_path):
                            img_str = generate_thumbnail(video_path)
                            thumbnail_html = f'<img src="data:image/png;base64,{img_str}" width="100" height="75">'
                        else:
                            thumbnail_html = "Thumbnail Not Available"

                        sorted_captions = sorted(captions, key=lambda x: x.get('elo_rating', 0), reverse=True)
                        for caption in sorted_captions[:top_n]:
                            html_report += f"        <tr>\n"
                            html_report += f"            <td>{thumbnail_html}</td>\n"
                            html_report += f"            <td>{video_id}</td>\n"
                            html_report += f"            <td>{caption['text']}</td>\n"
                            html_report += f"            <td>{caption.get('elo_rating', 0)}</td>\n"
                            html_report += f"        </tr>\n"

                    html_report += "    </tbody>\n"
                    html_report += "</table>\n"

                    return html_report


                refresh_stats_btn.click(
                    fn=generate_stats_report,
                    outputs=stats_report
                )

        return demo

def main():
    videos_dir = './videos'
    captions_dir = './captions_all'
    caption_best_dir = './captions_best'
    app = VideoCaptionApp(videos_dir, captions_dir, caption_best_dir)
    interface = app.create_gradio_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
