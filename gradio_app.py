import gradio as gr
from theme_classifier.theme_classifier import ThemeClassifier
from character_network import CharacterNetworkGenerator, NamedEntityRecognizer


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

def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def main():
    with gr.Blocks() as iface:

        # Theme Classification Section

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

        # Character Network Section

        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script path")
                        ner_path = gr.Textbox(label="NERs Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network,
                                                inputs=[subtitles_path, ner_path], outputs=[network_html])

    iface.launch(share=True)

if __name__ == '__main__':
    main()