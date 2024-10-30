import os
import re
import csv
import uuid
import glob
import shutil

# Variables
source_dir = "source_files/"

table_output_dir = "extracted_tables/"
table_code_dir = table_output_dir + "table_code/"
table_header_dir = table_output_dir + "table_header/"
table_result_file = "tables_regex.csv"

figure_output_dir = "extracted_figures/"
figure_result_file = "figures_regex.csv"

possible_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".eps"]

os.makedirs(table_output_dir, exist_ok=True)
os.makedirs(table_header_dir, exist_ok=True)
os.makedirs(table_code_dir, exist_ok=True)
os.makedirs(figure_output_dir, exist_ok=True)

# Reset directories
files = glob.glob(table_code_dir + "*")
for f in files:
    os.remove(f)

files = glob.glob(table_header_dir + "*")
for f in files:
    os.remove(f)

files = glob.glob(figure_output_dir + "*")
for f in files:
    os.remove(f)

if os.path.isfile(table_output_dir + table_result_file):
    os.remove(table_output_dir + table_result_file)

if os.path.isfile(figure_output_dir + figure_result_file):
    os.remove(figure_output_dir + figure_result_file)

csvfile_table = open(table_output_dir + table_result_file, 'w', newline='', encoding="utf-8")
spamwriter_table = csv.writer(csvfile_table, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

csvfile_figure = open(figure_output_dir + figure_result_file, 'w', newline='', encoding="utf-8")
spamwriter_figure = csv.writer(csvfile_figure, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

for paper in os.listdir(source_dir):
    paper_path = source_dir + paper
    if os.path.isdir(paper_path):
        print("\n\n" + paper)
        tex_files = [x for x in os.listdir(paper_path) if x.endswith('.tex')]

        complete_tex = ""
        for file in tex_files:
            try:
                f = open(paper_path + "/" + file, "r", encoding="utf8")
                complete_tex += f.read()
                f.close()
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError occurred. File could not be loaded.")
                continue

        found_document_header = re.search(r"\\documentclass\[.*\\begin\{document\}", complete_tex, re.DOTALL)
        paper_id = str(uuid.uuid4())
        if found_document_header:
            paper_header_file = table_header_dir + paper_id + ".txt"
            f = open(paper_header_file, "w", encoding="utf-8")
            f.write(found_document_header.group(0))
            f.close()
        else:
            print("Document header could not be identified.")
            continue

        found_tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", complete_tex, re.DOTALL)
        found_figures = re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", complete_tex, re.DOTALL)

        for table in found_tables:
            # r"\\caption\{(.*?)}"
            #caption_match = re.search(r"\\caption\{(([^{}]*(\{[^{}]*\})?[^{}]*)+)\}", table)
            caption_match = re.search(r"\\caption\{(.*?)\}", table)
            if caption_match:
                found_caption = caption_match.group(1)
                table_id = str(uuid.uuid4())

                spamwriter_table.writerow([table_id, paper_id, found_caption])
                table_file_path = table_code_dir + table_id + ".txt"
                f = open(table_file_path, "w", encoding="utf-8")
                f.write(table)
                f.close()

        for figure in found_figures:
            found_graphics = re.findall(r"\\includegraphics(\[.*?\])*\{(.*?)\}", figure)
            caption_match = re.search(r"\\caption\{(.*?)\}", figure)
            #caption_match = re.search(r"\\caption\{(([^{}]*(\{[^{}]*\})?[^{}]*)+)\}", figure)
            if caption_match:
                found_caption = caption_match.group(1)
                for graphic in found_graphics:
                    figure_id = str(uuid.uuid4())

                    graphic_path = graphic[1]
                    file_type = os.path.splitext(graphic_path)[-1]
                    if os.path.isfile(paper_path + "/" + graphic_path):
                        graphic_path = paper_path + "/" + graphic_path
                    else:
                        for ext in possible_extensions:
                            possible_path = paper_path + "/" + graphic_path + ext
                            if os.path.isfile(possible_path):
                                graphic_path = possible_path
                                break

                    try:
                        with open(graphic_path, "rb") as file_original:
                            content = file_original.read()
                            with open(figure_output_dir + figure_id + file_type, "wb") as file_copy:
                                file_copy.write(content)
                        # shutil.copy(graphic_path, figure_output_dir + figure_id + file_type)
                        spamwriter_figure.writerow([figure_id, paper_id, found_caption])
                    except FileNotFoundError as e:
                        print(f"File not found: {graphic_path} - {e}")
                    except Exception as e:
                        print(f"An error occurred: {e}")

        print(f"{len(found_tables)} tables found.")
        print(f"{len(found_figures)} figures found.")
    #break

csvfile_table.close()
csvfile_figure.close()