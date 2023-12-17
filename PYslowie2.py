import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QCheckBox, QProgressBar, QTextEdit
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

MAX_LENGTH = 200

class ProverbsSearchApp(QWidget):
    def __init__(self):
        super().__init__()

        self.proverbs = None
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

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
        layout.addWidget(self.progress_bar)

        layout.addWidget(label)
        layout.addWidget(self.entry)
        layout.addWidget(search_button)
        layout.addWidget(self.results_label)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.generate_button)

        self.generated_text_edit = QTextEdit()
        self.save_button = QPushButton("Save as TXT")
        self.save_button.clicked.connect(self.save_generated_text)
        self.save_button.setEnabled(False)

        layout.addWidget(self.generated_text_edit)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.setWindowTitle("Search Proverbs")
        self.setGeometry(100, 100, 600, 400)

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

        # Use AI model to generate a new proverb based on selected proverbs
        generated_proverb = self.ai_generate(selected_proverbs)
        self.generated_text_edit.setPlainText(generated_proverb)

        # Enable the Save as TXT button
        self.save_button.setEnabled(True)

    def ai_generate(self, selected_proverbs):
        input_text = "\n".join(selected_proverbs)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        generated_output = self.model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True)

        # Continuously generate text until reaching a specified length
        while generated_output.size(1) < MAX_LENGTH:
            # Adjust the max_length parameter dynamically based on the current output length
            current_max_length = min(MAX_LENGTH, generated_output.size(1) + 50)  # Increase by 50 tokens in each iteration
            output = self.model.generate(generated_output, max_length=current_max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
            generated_output = torch.cat([generated_output, output[:, -50:]], dim=1)  # Concatenate the last 50 tokens to the generated output

            # Update progress bar
            progress_value = int((generated_output.size(1) / MAX_LENGTH) * 100)
            self.progress_bar.setValue(progress_value)

        decoded_output = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
        return decoded_output

    def save_generated_text(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Generated Text", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_name:
            with open(file_name, 'w', encoding='utf-8') as file:
                generated_text = self.generated_text_edit.toPlainText()
                file.write(generated_text)
            print("Generated text saved to:", file_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ProverbsSearchApp()
    window.show()
    sys.exit(app.exec_())
