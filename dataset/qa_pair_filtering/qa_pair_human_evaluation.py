import os
import csv
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import font

# Variables and directories
qa_pair_file = "dataset/test_split.csv"
positive_output_file = "dataset/qa_pairs_accepted.csv"
negative_output_file = "dataset/qa_pairs_rejected.csv"
image_path_tables = "dataset/extracted_tables/table_images/"
image_path_figures = "dataset/extracted_figures/"

# Defining column indexes
object_id_index = 0
object_type_index = 1
question_index = 2
answer_index = 3

# Create output files
if not os.path.exists(positive_output_file):
    with open(positive_output_file, "w"):
        pass
if not os.path.exists(negative_output_file):
    with open(negative_output_file, "w"):
        pass

def get_set_of_evaluated_questions():
    """
    This functions iterates through the output files to construct a set of questions which were already evaluated by a human.

    Returns:
        set[str]: A set of already evaluated questions.
    """
    output_files = [positive_output_file, negative_output_file]
    question_set = set()

    # Iterate through output files to construct set
    for output_f in output_files:
        if os.path.getsize(output_f) != 0:
            df = pd.read_csv(output_f, delimiter=";", quotechar="|", header=None)
            df = df.iloc[:, question_index]

            for question in df:
                question_set.add(question)

    return question_set

def identify_lowest_index():
    """
    This function identifies the index of the first QA-pair that has not been yet evaluated by a human.

    Returns:
        int: The index of the first unevaluated QA-pair in qa_pair_file.
    """
    question_set = get_set_of_evaluated_questions()
    lowest_index = 0

    # Iterate through csv file to locate index
    with open(qa_pair_file, "r", encoding="utf-8") as input_f:
        csv_reader = csv.reader(input_f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for row in csv_reader:
            if row[question_index] in question_set:
                lowest_index += 1
            else:
                break

    return lowest_index

def update_display():
    """
    This function updates the image and text labels in the GUI after a QA-pair has been evaluated.
    """
    global current_index, img_label, img

    # Retrieve data
    row = data.iloc[current_index]
    object_id = row[object_id_index]
    object_type = row[object_type_index]
    question = row[question_index]
    answer = row[answer_index]

    # Update index label
    index_label.config(text=f"Index: {current_index} / {len(data)}")

    # Clear previous text
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)

    # Insert question and answer
    text_widget.insert(tk.END, "Question: ", "bold")
    text_widget.insert(tk.END, question + "\n\n")
    text_widget.insert(tk.END, "Answer: ", "bold")
    text_widget.insert(tk.END, answer + "\n")
    text_widget.config(state="disabled")

    # Determine image path
    image_path = None
    if object_type == "Figure":
        image_path = image_path_figures + object_id + ".png"
    else:
        image_path = image_path_tables + object_id + ".png"

    if os.path.exists(image_path):
        img = Image.open(image_path)
        img.thumbnail((500,500))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
    else:
        img_label.config(text=f"Image not found: {image_path}", image='')

def classify(positive_class):
    """
    This function is called when the user decided whether a QA-pair should be kept or dismissed.
    The QA-pair will be written to a corresponding csv file, dependent of the user's decision.
    Afterwards, the index will be updated and next QA-pair will be loaded.

    Args:
        positive_class (bool): If True, the current QA-pair shall remain included in the test set.
    """
    global current_index
    
    row = data.iloc[current_index]
    output_file = positive_output_file if positive_class else negative_output_file

    with open(output_file, "a", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

    current_index += 1
    update_display()

# Load csv and start index
data = pd.read_csv(qa_pair_file, delimiter=";", quotechar="|", header=None)
current_index = identify_lowest_index()

# GUI setup
root = tk.Tk()
root.title("QA-Pair Human Evaluation")

# Text and image
text_widget = tk.Text(root, wrap="word", font=("Arial", 11), height=7, width=70, state="disabled")
text_widget.pack(pady=10, fill="both", expand=True)

bold_font = font.Font(family="Arial", size=11, weight="bold")
text_widget.tag_configure("bold", font=bold_font)

img_label = tk.Label(root)
img_label.pack()

index_label = tk.Label(root, text="", wraplength=400, pady=2)
index_label.pack()

# Buttons
accept_btn = tk.Button(root, text="Accept", command=lambda: classify(True))
accept_btn.pack(side="left", padx=20, pady=5)
reject_btn = tk.Button(root, text="Reject", command=lambda: classify(False))
reject_btn.pack(side="right", padx=20, pady=5)

# Key binding
root.bind("1", lambda event: classify(True))
root.bind("2", lambda event: classify(False))

update_display()
root.mainloop()