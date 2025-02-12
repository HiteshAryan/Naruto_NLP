import gradio as gr
from theme_classifier.theme_classifier import ThemeClassifier


def get_themes(themes_list_str, subtitles_path, save_path):
    themes_list = themes_list_str.split(',')
    theme_classifier = ThemeClassifier(themes_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove dialogue from theme list
    themes_list = [theme for theme in themes_list if theme != 'dialogue']
    output_df = output_df[themes_list]

    output_df = output_df[themes_list].sum().reset_index()
    output_df.columns = ['theme', 'score']

    output_chart = gr.BarPlot(
        output_df,
        x='score',
        y='theme',
        title='Series Themes',
        tooltip=['theme', 'score'],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or Script path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes,
                                                inputs=[theme_list, subtitles_path, save_path], outputs=[plot])

    iface.launch(share=True)

if __name__ == '__main__':
    main()