import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog

def read_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.lower() for line in file.readlines()]

def search_proverbs(query, proverbs):
    query = query.lower()
    results = {}

    for i, proverb in enumerate(proverbs):
        if query in proverb:
            results[i] = proverb

    unique_results = list(set(results.values()))
    return unique_results[:10]

class ProverbsSearchApp(QWidget):
    def __init__(self):
        super().__init__()

        self.proverbs = None

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

        layout.addWidget(label)
        layout.addWidget(self.entry)
        layout.addWidget(search_button)
        layout.addWidget(self.results_label)
        layout.addWidget(self.browse_button)

        self.setLayout(layout)
        self.setWindowTitle("Search Proverbs")
        self.setGeometry(100, 100, 400, 200)

    def on_search_clicked(self):
        if self.proverbs is None:
            self.results_label.setText("Please select a database file.")
            return

        query = self.entry.text()
        results = search_proverbs(query, self.proverbs)
        self.display_results(results)

    def display_results(self, results):
        if results:
            result_text = "Top 10 matching proverbs:\n"
            for i, proverb in enumerate(results):
                result_text += f"{i + 1}. {proverb.strip()}\n"
        else:
            result_text = "No matching proverbs found."

        self.results_label.setText(result_text)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ProverbsSearchApp()
    window.show()
    sys.exit(app.exec_())
