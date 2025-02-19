PROMPT_TEMPLATE_TASK1 = """
Answer the following question regarding a scientific {object} in a short way:
{object} caption: {caption}
{table_code_filler}
Question: {question}
"""

PROMPT_TEMPLATE_TASK23 = """
Answer the following single choice question by providing the correct letter (A, B, C, or D).
Provide only the letter corresponding to the correct answer.

Question: {question}
A) {option1}
B) {option2}
C) {option3}
D) {option4}
"""

QUESTION_TASK2 = "To which of these paper title does the following {object} belong?"
QUESTION_TASK3 = "Which of these text passages reference the following {object}?"
