import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QCheckBox, QProgressBar, QTextEdit
from transformers import AutoTokenizer, AutoModelForCausalLM
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

MAX_LENGTH = 1000

class ProverbsSearchApp(QWidget):
    def __init__(self):
        super().__init__()

        self.proverbs = None

        # Model GPT-2
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/zabanshenas-roberta-base-mix")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("m3hrdadfi/zabanshenas-roberta-base-mix")

        # Model Word2Vec
        model_path = hf_hub_download(repo_id="Word2vec/nlpl_62", filename="model.bin")
        self.word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        label = QLabel("Enter your question:")
        self.entry = QLineEdit()
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.on_search_clicked)

        self.results_label = QLabel("")

        self.browse_button = QPushButton("Browse Database")
        self.browse_button.clicked.connect(self.on_browse_clicked)

        self.generate_button = QPushButton("Generate Proverb")
        self.generate_button.clicked.connect(self.generate_proverb)
        self.generate_button.setEnabled(False)

        self.progress_bar = QProgressBar()

        # Text area to display generated proverb
        self.generated_text_area = QTextEdit()
        self.generated_text_area.setReadOnly(True)

        layout.addWidget(self.progress_bar)
        layout.addWidget(label)
        layout.addWidget(self.entry)
        layout.addWidget(search_button)
        layout.addWidget(self.results_label)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.generated_text_area)

        self.setLayout(layout)
        self.setWindowTitle("Search Proverbs")
        self.setGeometry(100, 100, 400, 400)

    def on_search_clicked(self):
        if self.proverbs is None:
            self.results_label.setText("Please select a database file.")
            return

        query = self.entry.text()
        results = self.search_proverbs(query)
        self.display_results(results)

    def display_results(self, results):
        checkboxes = []
        if results:
            result_text = "Matching proverbs:\n"
            for i, proverb in enumerate(results):
                checkbox = QCheckBox(proverb.strip())
                checkboxes.append(checkbox)
                self.layout().insertWidget(i + 4, checkbox)  # Insert checkboxes after the results_label
                result_text += f"{i + 1}. {proverb.strip()}\n"

        else:
            result_text = "No matching proverbs found."

        self.results_label.setText(result_text)

        # Enable the Generate Proverb button only if there are results
        self.generate_button.setEnabled(bool(results))

    def on_browse_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select Database File", "", "Text Files (*.txt);;All Files (*)", options=options)

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    self.proverbs = [line.lower() for line in file.readlines()]
                    print("Loaded proverbs:", self.proverbs)
                    self.results_label.setText("Database selected: " + filename)
            except Exception as e:
                print("Error loading database:", str(e))
                self.results_label.setText("Error loading database.")

    def search_proverbs(self, query):
        query = query.lower()
        results = {}

        for i, proverb in enumerate(self.proverbs):
            if query in proverb:
                results[i] = proverb

        unique_results = list(set(results.values()))
        return unique_results[:10]

    def generate_proverb(self):
        if self.proverbs is None:
            return

        selected_proverbs = [checkbox.text() for checkbox in self.findChildren(QCheckBox) if checkbox.isChecked()]
        if not selected_proverbs:
            return

        # Use GPT-2 model to generate a new proverb based on selected proverbs
        generated_proverb = self.gpt2_generate(selected_proverbs)
        self.generated_text_area.setPlainText(f"Generated Proverb: {generated_proverb}")

    def gpt2_generate(self, selected_proverbs):
        input_text = "\n".join(selected_proverbs)
        input_ids = self.gpt2_tokenizer.encode(input_text, return_tensors="pt")

        generated_output = self.gpt2_model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        generated_proverb = self.gpt2_tokenizer.decode(generated_output[0], skip_special_tokens=True)
        return generated_proverb


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProverbsSearchApp()
    window.show()
    sys.exit(app.exec_())
